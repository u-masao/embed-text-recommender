"""
このモジュールは DummySearchEngine モデルを実装します
"""
from typing import List, Tuple

import numpy as np

from .search_engine import SearchEngineStrategy


class DummySearchEngine(SearchEngineStrategy):
    """
    ダミーの実装です。ベクトルDBに対する検索エンジン。
    """

    def __init__(
        self,
        dimension: int = 10,
        random_seed: int = 1234,
    ):
        """
        SearchEngine を初期化する。

        Parameters
        ------
        dimension: int
            モデルの次元
        random_seed: int
            ランダムシード

        Returns
        ------
        None
        """
        self.dimension = dimension
        self.random_seed = random_seed
        self.vector_store = None

    def get_embed_dimension(self):
        """
        Embedding の次元数を返す

        Parameters
        ------
        None

        Returns
        ------
        int
            モデルの次元数
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
            埋め込みベクトル行列または埋め込みベクトル

        Returns
        ------
        None
        """
        pass

    def ids_to_embeds(self, ids: List[int]) -> np.ndarray:
        """
        ID を渡すと Embeddings を返す。

        Parameters
        ------
        ids: List[int]
            ID のリスト

        Returns
        ------
        np.ndarray
            埋め込みベクトル行列
        """
        n_size = len(ids)
        np.random.seed(self.random_seed)
        embeds = np.random.standard_normal((n_size, self.dimension))
        embeds = embeds / np.linalg.norm(embeds, axis=0, ord=2)
        return embeds

    def search(self, query_embeds, top_n=5) -> Tuple[np.ndarray, np.ndarray]:
        """
        入力クエリ埋め込みベクトルに近いベクトルを検索する。
        一つのベクトルでも複数のベクトルでも動作する。
        top_n に満たない件数を返す場合はない。

        Parameters
        ------
        query_embeds: np.ndarray
            クエリとなる埋め込みベクトル行列または埋め込みベクトル
        top_n: int
            返す件数

        Returns
        ------
        np.ndarray
            検索結果 ID の行列
        np.ndarray
            距離（類似度）の行列
        """

        np.random.seed(self.random_seed)
        if query_embeds.ndim == 1:
            result_ids = np.random.randint(1, high=1000, size=(1, top_n))
            result_distances = np.random.randn(1, top_n)
        elif query_embeds.ndim == 2:
            result_ids = np.random.randint(
                1, high=1000, size=(query_embeds.shape[0], top_n)
            )
            result_distances = np.random.randn(query_embeds.shape[0], top_n)
        else:
            raise ValueError("入力 query_embeds のサイズは (d,) または (n,d) として下さい。")
        return result_ids, result_distances

    def save(self, filepath: str):
        """
        インスタンスをファイルに保存する。
        """

        pass
