import logging
from pprint import pformat
from typing import List

import numpy as np
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm.contrib import tmap


class Embedder:
    def __init__(
        self, model_name_or_filepath, chunk_overlap=50, tokens_par_chunk=None
    ):
        self.model_name_or_filepath = model_name_or_filepath
        self.model = SentenceTransformer(model_name_or_filepath)
        self.splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            model_name=model_name_or_filepath,
            tokens_per_chunk=tokens_par_chunk,
        )

    def encode(self, sentences: List[str], method: str = "chunk_split"):
        """
        埋め込みを計算する。

        Parameters
        ------
        sentences: List[str]
            センテンスの List
        method: str
            埋め込みの計算方法を指定する。以下に対応。
            - chunk_split (default)
              - 各センテンスをチャンクに分割し埋め込みを計算する。
              - 埋め込みの平均ベクトルを返す。
            - head_only
              - チャンクに分割せずに埋め込みモデルで処理する。
              - モデルの max_tokens のみの埋め込みを計算する

        Returns
        ------
        numpy.ndarray
            埋め込み行列。
            センテンスの数 n、埋め込み次元 d とすると、(n, d) の2次元ベクトルを返す。
        """
        # init logger
        logger = logging.getLogger(__name__)
        logger.info(f"encode method: {method}")

        # それぞれの方法で embedding する
        if method == "head_only":
            embeddings = self._head_only_encode(sentences)
        elif method == "naive_chunk_split":
            embeddings = self._naive_split_encode(sentences)
        elif method == "chunk_split":
            embeddings = self._make_chunk_averaged_encode(sentences)
        else:
            ValueError(f"指定の method には対応していません。 method: {method}")

        return embeddings

    def _head_only_encode(self, sentences):
        return self.model.encode(sentences)

    def _naive_split_encode(self, sentences):
        embeddings = []
        for sentence in sentences:
            vectors = self.model.encode(self.splitter.split_text(sentence))
            mean_vector = np.mean(vectors, axis=0)
            assert (
                len(mean_vector)
                == self.model.get_sentence_embedding_dimension()
            )
            embeddings.append(mean_vector)
        result = np.array(embeddings)
        assert result.shape[0] == len(sentences)
        assert result.shape[1] == self.model.get_sentence_embedding_dimension()
        return result

    def _make_chunk_averaged_encode(self, sentences: List[str]) -> np.ndarray:
        """
        センテンス毎のチャンク List を受け取り、センテンス毎の
        平均埋め込みベクトルを返す。

        Parameters
        ------
        sentences: List[str]
            センテンスのリスト

        Returns
        ------
        numpy.ndarray
            センテンス毎の埋め込み表現
            次元は、センテンス数 n、モデル次元 d に対して、(n, d)となる。
        """

        # init logger
        logger = logging.getLogger(__name__)
        d_size = self.model.get_sentence_embedding_dimension()

        # split to chunks
        logger.info("split sentences to chunks")
        chunks_list = [
            x
            for x in tmap(
                lambda x: self.splitter.split_text(text=x), sentences
            )
        ]

        # チャンクを 1 次元の List に flatten する
        chunk_list = flatten_chunks(chunks_list)

        # matrix mask を作成
        logger.info("make weight matrix")
        weight_matrix = make_weight_matrix(chunks_list)
        assert weight_matrix.shape[0] == len(chunks_list)
        assert weight_matrix.shape[1] == len(chunk_list)

        # 埋め込みを計算
        logger.info("encode chunks")
        chunk_embeddings = self.model.encode(chunk_list)
        assert chunk_embeddings.shape[0] == weight_matrix.shape[1]
        assert chunk_embeddings.shape[1] == d_size

        # ウェイト行列とチャンク毎のEmbeddingで行列積を計算
        logger.info("calc dot matrix, weight_matrix @ chunk_embeddings")
        embeddings = np.dot(weight_matrix, chunk_embeddings)
        assert embeddings.shape[0] == len(chunks_list)
        assert embeddings.shape[1] == d_size

        return embeddings

    def __str__(self) -> str:
        params = {
            "model": str(self.model),
            "indexer": str(self.splitter),
            "model_name_or_path": self.model_name_or_path,
        }
        return pformat(params)


def make_weight_matrix(chunks_list: List[List[str]]) -> np.ndarray:
    """
    チャンク分割された文字列を結合するための行列を作る。
    オリジナルのテキスト数が n_size となる。
    チャンク分割されたテキスト数が c_size となる。

    Parameters
    ------
    chunks_list: List[List[str]]
        チャンク分割された文字列。それぞれの要素内の要素数が異なる。

    Returns
    ------
    numpy.ndarray
        結合するためのマスクを返す。
    """

    # 出力の次元を計算する
    n_size = len(chunks_list)
    c_size = np.sum([x for x in map(lambda x: len(x), chunks_list)])

    # 出力する変数を初期化
    weight_matrix = np.zeros((n_size, c_size))

    # offset を定義
    chunk_offset = 0

    # 各オリジナルテキストの各チャンクでループ
    for n_index, chunks in enumerate(chunks_list):
        for c_index, chunk in enumerate(chunks):
            # Mask Matrix に重みを代入
            weight_matrix[n_index, c_index + chunk_offset] = 1 / len(chunks)
        chunk_offset += len(chunks)

    # サイズチェックと値チェック
    assert weight_matrix.sum() == n_size
    assert np.mean(weight_matrix.sum(axis=1) ** 2) == 1

    return weight_matrix


def flatten_chunks(chunks_list: List[List[str]]) -> List[str]:
    """
    チャンクを一次元の List に再配置する。
    chunks_list -> chunk_list

    Parameters
    ------
    chunks_list: List[List[str]]
        配列内の配列としてチャンクを保持するデータ

    Returns
    ------
    List[str]
        チャンクの List
    """
    chunk_list = []
    for chunks in chunks_list:
        for chunk in chunks:
            chunk_list.append(chunk)
    return chunk_list
