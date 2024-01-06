"""
このモジュールは FaissSearchEngine モデルを実装します
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import cloudpickle
import faiss
import numpy as np

from .search_engine import SearchEngineStrategy


class FaissSearchEngine(SearchEngineStrategy):
    """
    FAISS を利用したベクトル検索エンジンクラス
    """

    def __init__(
        self,
        dimension: int,
        metric_space_distance: str = "cosine",
        enable_gpu: bool = False,
    ):
        """
        VectorEngine を初期化する。

        Parameters
        ------
        dimension: int
            埋め込みの次元数
        metric_space_distance: str
            距離空間の定義。以下の値に対応する。
            - コサイン類似度（dot & normalize）
              - cosine
            - ユークリッド距離
              - euclidean
            - 正規化なし内積
              - dot
            - 正規化済みユークリッド距離
              - normalized_euclidean
            デフォルト値は cosine
        enable_gpu: bool
            GPU を利用する場合のフラグ

        Returns
        ------
        None
        """

        # インスタンス変数を保存
        self.dimension = dimension
        self.metric_space_distance = metric_space_distance
        self.enable_gpu = enable_gpu

        # Faiss を初期化
        self.index = self._initialize_faiss_index(
            dimension, metric_space_distance, enable_gpu
        )

    def _initialize_faiss_index(
        self, dimension, metric_space_distance, enable_gpu
    ):
        """
        Faiss を初期化する。

        Parameters
        ------
        dimension: int
            埋め込みの次元数
        metric_space_distance: str
            距離空間の定義。以下の値に対応する。
            - コサイン類似度（dot & normalize）
              - cosine
            - ユークリッド距離
              - euclidean
            - 正規化なし内積
              - dot
            - 正規化済みユークリッド距離
              - normalized_euclidean
            デフォルト値は cosine
        enable_gpu: bool
            GPU を利用する場合のフラグ

        Returns
        ------
        None
        """
        # オプションをパース
        if metric_space_distance == "cosine":
            self.enable_compare_normalize = True
            self.index_type = "IP"
        elif metric_space_distance == "euclidean":
            self.enable_compare_normalize = False
            self.index_type = "L2"
        elif metric_space_distance == "dot":
            self.enable_compare_normalize = False
            self.index_type = "IP"
        elif metric_space_distance == "normalized_euclidean":
            self.enable_compare_normalize = True
            self.index_type = "L2"
        else:
            raise ValueError("指定の metric_space_distance はサポートしていません")
        if enable_gpu:
            if self.index_type == "IP":
                # 内積距離空間を初期化
                base_index = faiss.GpuIndexFlatIP(dimension)
            elif self.index_type == "L2":
                # Euclid 距離空間を初期化
                base_index = faiss.GpuIndexFlatL2(dimension)
            # ID map でラップ
            index = faiss.GpuIndexIDMap(base_index)
        else:
            if self.index_type == "IP":
                # 内積距離空間を初期化
                base_index = faiss.IndexFlatIP(dimension)
            elif self.index_type == "L2":
                # Euclid 距離空間を初期化
                base_index = faiss.IndexFlatL2(dimension)
            # ID map でラップ
            index = faiss.IndexIDMap(base_index)
        return index

    def get_embed_dimension(self) -> Optional[int]:
        """
        Embedding の次元数を返す

        Parameters
        ------
        None

        Returns
        ------
        Optional[int]
            モデルの次元数。ベクトルを追加する前は None を返す
        """
        return self.dimension

    def add_ids_and_embeds(self, ids: List[int], embeds: np.ndarray) -> None:
        """
        ID と Embeddings をインスタンス内部のベクトルストアに保存する。

        Parameters
        ------
        ids: List[int]
            ID のリスト
        embeds: np.ndarray
            埋め込みベクトル行列

        Returns
        ------
        None
        """

        if embeds.ndim > 2 or self.dimension != embeds.shape[-1]:
            raise ValueError("入力 embeds が正しくありません")

        # (n, self.dimension) shape に整形
        embeds = embeds.reshape(-1, self.dimension)

        # インスタンス変数を初期化
        self.ids = [int(x) for x in list(ids)]
        self.embeds = embeds

        # normalize vectors
        l2norms = np.linalg.norm(embeds, axis=-1, ord=2)
        self.normalized_embeds = (embeds.T / l2norms).T

        if self.enable_compare_normalize:
            # 正規化ベクトル（行列）を追加
            self.index.add_with_ids(self.normalized_embeds, ids)
        else:
            # 生のベクトル（行列）を追加
            self.index.add_with_ids(self.embeds, ids)

    def ids_to_embeds(
        self, ids: List[int], normalized: bool = False
    ) -> np.ndarray:
        """
        ID に対して埋め込み表現を返す。

        Parameters
        ------
        ids: List[int]
            クエリー ID。
        normalized: bool
            正規化ベクトルを返すオプション。

        Returns
        ------
        np.ndarray
            埋め込み行列
        """
        if normalized:
            return self.normalized_embeds[self._find_ids_indices(ids)]
        else:
            return self.embeds[self._find_ids_indices(ids)]

    def search(
        self, query_embeds: np.ndarray, top_n=5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        クエリーベクトルに対して、距離が近いベクトルを計算する。
        IDとコサイン類似度を返す。
        一つのベクトルでも複数のベクトルでも動作する。
        top_n に満たない件数を返す場合はない。

        Parameters
        ------
        query_embeds: np.ndarray
            クエリとなる埋め込みベクトル行列または埋め込みベクトル
            以下の shape に対応。
            - (self.dimension,)
            - (n, self.dimension) -> 複数の結果を返す
        top_n: int
            返却する件数。

        Returns
        ------
        numpy.ndarray
            ID一覧。以下の shape で返す。
            - (top_n)
            - (n, top_n)
        numpy.ndarray
            距離のベクトル。以下の shape で返す。
            - (top_n)
            - (n, top_n)
        """

        # (n, self.dimension) shape に整形
        query_embeds = query_embeds.reshape(-1, self.dimension)

        if self.enable_compare_normalize:
            # クエリーベクトルの長さを計算し正規化
            l2norms = np.linalg.norm(query_embeds, axis=1, ord=2)
            search_query_embeds = (query_embeds.T / l2norms).T
        else:
            # クエリーベクトルはそのまま利用
            search_query_embeds = query_embeds

        # 検索
        similarities, indices = self.index.search(search_query_embeds, top_n)

        return indices, similarities

    def _find_ids_indices(self, ids: List[int]):
        """
        ID から Engine に登録されている id を抽出。

        Parameters
        ------
        ids: List[int]
            クエリー ID。

        Returns
        ------
        List[int]
            存在するクエリー ID のリスト。
        """
        results = []
        for id in ids:
            if id in self.ids:
                results.append(self.ids.index(id))
        return results

    def save(self, filepath: str):
        """
        インスタンスをファイルに保存する。

        Parameters
        ------
        filepath: str
            読み込むファイル名

        Returns
        ------
        None
        """
        Path(filepath).parent.mkdir(exist_ok=True, parents=True)
        cloudpickle.dump(self, open(filepath, "wb"))

    @classmethod
    def load(cls, filepath: str):
        """
        ファイルからインスタンスを読み込む。

        Parameters
        ------
        filepath: str
            書き込むファイル名

        Returns
        ------
        VectorEngine
        """
        return cloudpickle.load(open(filepath, "rb"))

    def __str__(self) -> str:
        params = {
            "indexer": str(self.index),
            "ids length": len(self.ids),
            "embeddings shape": self.embeds.shape,
            "normalized_embeddings shape": self.normalized_embeds.shape,
            "embedding dimension": self.dimension,
            "metric_space_distance": self.metric_space_distance,
        }
        return json.dumps(params)
