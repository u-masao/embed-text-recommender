"""
このモジュールは Embedding モデルを管理します
インタフェースを定義し、個別の EmbeddingModel を実装します。
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingStrategy(ABC):
    """
    Embedding アルゴリズムのインタフェースを定義します。
    """

    @abstractmethod
    def embed(self, sentences):
        """
        Embedding を計算し結果を返します。
        """
        pass

    @abstractmethod
    def get_embed_dimension(self):
        """
        Embedding model が出力する embed の次元を返します。
        """
        pass

    @abstractmethod
    def get_model_name(self):
        """
        Embedding model の名前を返します。モデル名など。
        """
        pass


class EmbeddingModel:
    """
    Strategy を利用した処理インタフェースを実装します。

    """

    def __init__(self, strategy: EmbeddingStrategy):
        self.strategy = strategy

    def set_strategy(self, strategy: EmbeddingStrategy):
        self.strategy = strategy

    def embed(self, sentences: List[str]) -> np.ndarray:
        return self.strategy.embed(sentences)

    def get_embed_dimension(self):
        return self.strategy.get_embed_dimension()

    def get_model_name(self):
        return self.strategy.get_model_name()

    def __str__(self):
        return self.strategy.__str__()

    @classmethod
    def make_embedding_model(cls, model_string: str, **kwargs):
        """
        EmbeddingModel インスタンスを返す
        """
        # ファイルの先頭に import を書くと循環参照になるため、利用時にインポートする
        from . import SentenceTransformerEmbedding, Word2VecEmbedding

        # 最初の / までが Storategy 名
        storategy = model_string.strip().split("/")[0]

        # 最初の / 以降が model_name_or_filepath
        name = "/".join(model_string.strip().split("/")[1:])

        # make EmbeddingStorategy
        if storategy == "SentenceTransformer":
            embedding_storategy = SentenceTransformerEmbedding(name, **kwargs)
        elif storategy == "Word2Vec":
            embedding_storategy = Word2VecEmbedding(name, **kwargs)
        else:
            raise ValueError(
                f"指定の embedding_storategy はサポートしていません: {storategy}"
            )

        # embedding
        return EmbeddingModel(embedding_storategy)
