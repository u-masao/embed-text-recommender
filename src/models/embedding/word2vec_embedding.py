"""
このモジュールは Word2VecEmbedding モデルを実装します
"""
from typing import List

import fugashi
import ipadic
import numpy as np
from gensim.models import KeyedVectors

from .embedding_model import EmbeddingStrategy


class Word2VecEmbedding(EmbeddingStrategy):
    """
    SentenceTransformer を利用した Embedding 実装です。
    """

    def __init__(
        self,
        model_name_or_filepath: str,
        **kwargs,
    ):
        """
        コンストラクタ。

        Parameters
        ------
        model_name_or_filepath: str
            埋め込みモデル名
        chunk_overlap: int
            チャンクオーバーラップトークン数
        tokens_par_chunk: Optional[int]
            チャンクあたりのトークン数。
            デフォルト値 None にすることで自動的にモデルの max_tokens を利用する。
        """
        self.model_name_or_filepath = model_name_or_filepath
        self.tagger = fugashi.GenericTagger(ipadic.MECAB_ARGS)
        self.model = KeyedVectors.load_word2vec_format(
            model_name_or_filepath, binary=True
        )

    def get_embed_dimension(self) -> int:
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
        return self.model.vector_size

    def get_model_name(self) -> str:
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
        return self.model_name_or_filepath

    def embed(self, sentences: List[str]):
        """
        埋め込みを計算する。

        Parameters
        ------
        sentences: List[str]
            センテンスの List

        Returns
        ------
        numpy.ndarray
            埋め込み行列。埋め込み次元 d とすると以下の行列やベクトルを返す。
            - センテンスの数 1
              - (d, ) の 1 次元ベクトル
            - センテンスの数 n (n>1)
              - (n, d) の行列
        """

        result = np.zeros((len(sentences), self.get_embed_dimension()))
        for index, sentence in enumerate(sentences):
            tags = self.tagger(sentence)
            for tag in tags:
                word = tag.surface
                if word in self.model:
                    result[index, :] += self.model[word]

            result[index, :] /= len(tags)

        return result

    def __str__(self) -> str:
        params = {
            "model": self.model,
            "model_name_or_filepath": self.model_name_or_filepath,
            "tagger": self.tagger,
        }
        return str(params)
