import logging
from typing import List

import numpy as np
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from sentence_transformers import SentenceTransformer
from tqdm.contrib import tmap


class VectorBuilder:
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

    def encode(self, sentences):
        # init logger
        logger = logging.getLogger(__name__)

        # split to chunks
        logger.info("split sentences to chunks")
        chunks_list = [
            x
            for x in tmap(
                lambda x: self.splitter.split_text(text=x), sentences
            )
        ]

        # embedding
        embeddings = make_chunk_averaged_embeddings(self.model, chunks_list)

        return embeddings

        # return self._naive_encode(sentences)

    def _naive_encode(self, sentences):
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


def make_chunk_averaged_embeddings(
    model: SentenceTransformer, chunks_list: List[List[str]]
) -> np.ndarray:
    """
    センテンス毎のチャンク List を受け取り、センテンス毎の
    平均埋め込みベクトルを返す。

    Parameters
    ------
    model: SentenceTransformer
        埋め込みモデル
    chunks_list: List[List[str]]
        センテンス毎のチャンク List

    Returns
    ------
    numpy.ndarray
        センテンス毎の埋め込み表現
        次元は、センテンス数 n、モデル次元 d に対して、(n, d)となる。
    """

    # init logger
    logger = logging.getLogger(__name__)
    d_size = model.get_sentence_embedding_dimension()

    # チャンクを 1 次元の List に flatten する
    chunk_list = flatten_chunks(chunks_list)

    # matrix mask を作成
    logger.info("make weight matrix")
    weight_matrix = make_weight_matrix(chunks_list)
    assert weight_matrix.shape[0] == len(chunks_list)
    assert weight_matrix.shape[1] == len(chunk_list)

    # 埋め込みを計算
    logger.info("encode chunks")
    chunk_embeddings = model.encode(chunk_list)
    assert chunk_embeddings.shape[0] == weight_matrix.shape[1]
    assert chunk_embeddings.shape[1] == d_size

    # ウェイト行列とチャンク毎のEmbeddingで行列積を計算
    logger.info("calc dot matrix, weight_matrix @ chunk_embeddings")
    embeddings = np.dot(weight_matrix, chunk_embeddings)
    assert embeddings.shape[0] == len(chunks_list)
    assert embeddings.shape[1] == d_size

    return embeddings


def test_main():
    import os

    import pandas as pd
    from tqdm import tqdm

    tqdm.pandas()

    # init model and splitter
    model_name = "intfloat/multilingual-e5-small"
    model = SentenceTransformer(model_name)
    splitter = SentenceTransformersTokenTextSplitter(model_name=model_name)

    # make dataset
    cache_filepath = "data/interim/tmp_cache_chunked.parquet"
    if os.path.isfile(cache_filepath):
        df = pd.read_parquet(cache_filepath)
    else:
        df = pd.read_parquet("data/interim/dataset.parquet")
        df["sentence"] = df["title"] + "\n" + df["content"]
        df["chunks"] = df["sentence"].progress_map(
            lambda x: splitter.split_text(text=x)
        )
        df.to_parquet(cache_filepath)

    embeddings = make_chunk_averaged_embeddings(model, df["chunks"])
    print(f"get embeds: {embeddings.shape}")


if __name__ == "__main__":
    import logging

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    test_main()
