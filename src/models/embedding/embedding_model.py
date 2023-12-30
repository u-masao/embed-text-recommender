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

    def embed(self, sentences: List[str], **kwargs) -> np.ndarray:
        return self.strategy.embed(sentences, **kwargs)

    def get_embed_dimension(self):
        return self.strategy.get_embed_dimension()
