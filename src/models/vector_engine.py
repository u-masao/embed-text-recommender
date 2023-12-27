from pathlib import Path
from typing import List

import cloudpickle
import faiss
import numpy as np


class VectorEngine:
    def __init__(self, ids, embeddings):
        """
        VectorEngine を初期化する。
        ID一覧と埋め込みを利用して内部のDBに登録する。

        Parameters
        ------
        ids: List[int]
            ID
        embeddings: numpy.ndarray
            埋め込み表現

        Returns
        ------
        None
        """

        # インスタンス変数を初期化
        self.ids = [int(x) for x in list(ids)]
        self.embeddings = embeddings
        self.dimension = embeddings.shape[1]

        # normalize vectors
        l2norms = np.linalg.norm(embeddings, axis=1, ord=2)
        self.normalized_embeddings = (embeddings.T / l2norms).T

        # 内積距離空間を初期化
        base_index = faiss.IndexFlatIP(self.dimension)

        # ID map でラップ
        self.index = faiss.IndexIDMap(base_index)

        # add vectors to faiss db
        self.index.add_with_ids(self.normalized_embeddings, ids)

    def search(self, query_embeddings, top_n: int = 5):
        """
        クエリーベクトルに対して、距離が近いベクトルを計算する。
        IDとコサイン類似度を返す。

        Parameters
        ------
        query_embeddings: numpy.ndarray(n, self.dimension)
            クエリーベクトル。以下の shape に対応。
            - (self.dimension,)
            - (n, self.dimension) -> 複数の結果を返す
        top_n: int
            返却する件数。

        Returns
        ------
        numpy.ndarray
            類似度のベクトル。以下の shape で返す。
            - (top_n)
            - (n, top_n)
        numpy.ndarray
            ID一覧。以下の shape で返す。
            - (top_n)
            - (n, top_n)
        """

        # (n, self.dimension) shape に整形
        query_embeddings = query_embeddings.reshape(-1, self.dimension)

        # クエリーベクトルの長さを計算し正規化
        l2norms = np.linalg.norm(query_embeddings, axis=1, ord=2)
        normalized_query_embeddings = (query_embeddings.T / l2norms).T

        # 検索
        similarities, indices = self.index.search(
            normalized_query_embeddings, top_n
        )

        return similarities, indices

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

    def ids_to_embeddings(self, ids: List[int], normalized: bool = False):
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
        numpy.ndarray
            埋め込み表現。
        """
        if normalized:
            return self.normalized_embeddings[self._find_ids_indices(ids)]
        else:
            return self.embeddings[self._find_ids_indices(ids)]

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
        Path(filepath).parent.mkdir(exist_ok=True)
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
