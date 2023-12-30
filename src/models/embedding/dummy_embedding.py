"""
このモジュールは DummyEmbedding モデルを実装します
"""
from typing import List

import numpy as np

from .embedding_model import EmbeddingStrategy


class DummyEmbedding(EmbeddingStrategy):
    """
    ダミーの Embedding 実装です。
    """

    def __init__(
        self,
        model_name: str = "random_embedding",
        dimension: int = 10,
        random_seed: int = 1234,
    ):
        """
        モデルを初期化する。

        Parameters
        ------
        model_name: str
            モデル名
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
        self.model_name = model_name

    def get_embed_dimension(self):
        """
        モデルの次元数を返す

        Parameters
        ------
        None

        Returns
        ------
        int
            モデルの次元数
        """
        return self.dimension

    def get_model_name(self):
        """
        モデルの次元数を返す

        Parameters
        ------
        None

        Returns
        ------
        str
            モデルの名前
        """
        return self.model_name

    def embed(self, sentences: List[str]) -> np.ndarray:
        """
        長さ 1 に正規化したランダムなベクトルを返します。

        Parameters
        ------
        sentences: List[str]
            文字列の List

        Returns
        ------
        np.ndarray
            ランダムな値のベクトル。行列のサイズは以下の通り。
            行数: 入力文字列の要素数 len(sentences)
            列数: モデル次元数 self.dimension
        """
        n_size = len(sentences)
        np.random.seed(self.random_seed)
        embeds = np.random.standard_normal((n_size, self.dimension))
        embeds = embeds / np.linalg.norm(embeds, axis=0, ord=2)

        return embeds
