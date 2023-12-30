from abc import ABC, abstractmethod
from typing import List, Tuple

import cloudpickle
import numpy as np


class SearchEngineStrategy(ABC):
    """
    SearchEngine のインタフェース定義クラス
    """

    @abstractmethod
    def get_embed_dimension(self):
        """
        埋め込みの次元数を返す
        """
        pass

    @abstractmethod
    def add_ids_and_embeds(self, ids: List[int], embeds: np.ndarray) -> None:
        """
        単数または複数のIDと埋め込みを SearchEngineStorategy 内に登録する
        """
        pass

    @abstractmethod
    def ids_to_embeds(self, ids: List[int]) -> np.ndarray:
        """
        単数または複数の ID から Embeds を取り出す
        """
        pass

    @abstractmethod
    def search(self, query_embeds, top_n=5) -> Tuple[List[int], np.ndarray]:
        """
        受け取った埋め込みベクトルに対して距離が近いIDと距離を返す。
        """
        pass

    @abstractmethod
    def save(self, filepath: str):
        """
        インスタンスをファイルに保存する。
        """
        pass


class SearchEngine:
    """
    IDと埋め込みを登録し、検索するクラス。SearchEngineStorategy を利用する。
    """

    def __init__(self, strategy: SearchEngineStrategy):
        """
        コンストラクタ。SearchEngineStrategy を登録する
        """
        self.strategy = strategy

    def get_embed_dimension(self):
        """
        埋め込みの次元数を返す
        """
        return self.strategy.get_embed_dimension()

    def add_ids_and_embeds(self, ids: List[int], embeds: np.ndarray) -> None:
        """
        単数または複数のIDと埋め込みを SearchEngineStorategy 内に登録する
        """
        self.strategy.add_ids_and_embeds(ids, embeds)

    def ids_to_embeds(self, ids: List[int], **kwargs) -> np.ndarray:
        """
        単数または複数の ID から Embeds を取り出す
        """
        return self.strategy.ids_to_embeds(ids, **kwargs)

    def search(self, query_embeds, top_n=5) -> Tuple[List[int], np.ndarray]:
        """
        受け取った埋め込みベクトルに対して距離が近いIDと距離を返す。
        """
        return self.strategy.search(query_embeds, top_n=top_n)

    def save(self, filepath: str):
        """
        インスタンスをファイルに保存する。
        """
        self.strategy.save(filepath)

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
        return cls(cloudpickle.load(open(filepath, "rb")))
