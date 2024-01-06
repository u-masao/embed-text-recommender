import logging

import numpy as np
import pandas as pd
import pytest

from src.models import QueryHandler
from src.models.embedding import EmbeddingModel
from src.models.search_engine import SearchEngine


@pytest.fixture(
    scope="session",
)
def engine(request):
    return SearchEngine.load(
        "models/SentenceTransformer/"
        "oshizo/sbert-jsnli-luke-japanese-base-lite"
        "/engine_limit-30_head_only.cloudpickle"
    )


@pytest.fixture(
    scope="session",
)
def embedding_model(request):
    return EmbeddingModel.make_embedding_model(
        "SentenceTransformer/oshizo/sbert-jsnli-luke-japanese-base-lite",
        chunk_method="chunk_split",
    )


@pytest.fixture(
    scope="session",
)
def query_handler(request):
    handler = QueryHandler()
    handler.init_models(
        "SentenceTransformer/oshizo/sbert-jsnli-luke-japanese-base-lite"
    )
    return handler


@pytest.fixture(
    scope="session",
)
def text_df(request):
    return pd.read_parquet("data/interim/sentences-limit-30.parquet")


def test_merge(embedding_model, engine, text_df, query_handler):
    # 初期化チェック
    assert embedding_model is not None
    assert engine is not None
    assert text_df is not None
    assert query_handler is not None

    # 適当な入力を設定
    positive_query = "就職活動 新卒"
    positive_query_blend_ratio = 1.1
    negative_query = "東京 東北"
    negative_query_blend_ratio = 1.1
    like_ids = " ".join(text_df["id"][:1].astype(str).tolist())
    like_blend_ratio = 1.1
    dislike_ids = " ".join(text_df["id"][2:4].astype(str).tolist())
    dislike_blend_ratio = 1.1

    # embedding
    positive_query_embeddings = embedding_query(
        positive_query, embedding_model
    )
    negative_query_embeddings = embedding_query(
        negative_query, embedding_model
    )

    # lookup
    like_embeddings = embedding_from_ids_string(like_ids, engine)
    dislike_embeddings = embedding_from_ids_string(dislike_ids, engine)

    # ベクトル合成
    total_embedding = query_handler.merge_embeddings(
        positive_query_embeddings,
        positive_query_blend_ratio,
        negative_query_embeddings,
        negative_query_blend_ratio,
        like_embeddings,
        like_blend_ratio,
        dislike_embeddings,
        dislike_blend_ratio,
    )

    d_size = embedding_model.get_embed_dimension()
    assert total_embedding.ndim == 1
    assert total_embedding.shape[0] == d_size


def test_single(embedding_model, engine, text_df, query_handler):
    # 初期化チェック
    assert embedding_model is not None
    assert engine is not None
    assert text_df is not None
    assert query_handler is not None

    positive_query = "就職活動"
    positive_query_blend_ratio = 0.9
    negative_query = ""
    negative_query_blend_ratio = 1.0
    like_ids = ""
    like_blend_ratio = 1.0
    dislike_ids = ""
    dislike_blend_ratio = 1.0

    positive_query_embeddings = embedding_query(
        positive_query, embedding_model
    )
    negative_query_embeddings = embedding_query(
        negative_query, embedding_model
    )
    like_embeddings = embedding_from_ids_string(like_ids, engine)
    dislike_embeddings = embedding_from_ids_string(dislike_ids, engine)

    # ベクトル合成
    total_embedding = query_handler.merge_embeddings(
        positive_query_embeddings,
        positive_query_blend_ratio,
        negative_query_embeddings,
        negative_query_blend_ratio,
        like_embeddings,
        like_blend_ratio,
        dislike_embeddings,
        dislike_blend_ratio,
    )

    # 差を計算
    diff_embedding = (
        total_embedding
        - (positive_query_embeddings * positive_query_blend_ratio).flatten()
    )

    d_size = embedding_model.get_embed_dimension()
    assert total_embedding.ndim == 1
    assert total_embedding.shape[0] == d_size
    assert positive_query_embeddings.ndim == 2
    assert positive_query_embeddings.shape == (1, d_size)
    assert np.linalg.norm(diff_embedding, ord=2) < 1e-10


def embedding_query(query, embedding_model):
    """
    検索クエリ文字列を埋め込みベクトルに変換する。
    """

    logger = logging.getLogger(__name__)

    # 結果変数を初期化
    query_embeddings = np.zeros(embedding_model.get_embed_dimension())

    # 入力文字列が空の場合はゼロベクトルを返す
    if len(query.strip()) == 0:
        return query_embeddings

    # 検索クエリをベクトル化
    sentences = split_text(query)
    query_embeddings = embedding_model.embed(sentences)
    logger.info(f"query_embed.shape: {query_embeddings.shape}")
    logger.info(
        "query_embeddings l2norm: "
        f"{np.linalg.norm(query_embeddings, axis=1, ord=2)}"
    )

    return query_embeddings


def embedding_from_ids_string(like_ids, engine):
    """
    id 一覧文字列をベクトルに変換する。
    """

    # Logger を初期化
    logger = logging.getLogger(__name__)

    # 結果のベクトルを初期化
    like_embeddings = np.zeros(engine.get_embed_dimension())

    # ID 一覧が空の場合は ゼロベクトルを返す
    if len(like_ids.strip()) == 0:
        return like_embeddings

    # int のリストに変換 (変換に失敗すると ValueError が出る)
    like_ids = [int(x) for x in split_text(like_ids)]
    logger.info(f"found like_ids: {like_ids}")

    # 埋め込みを検索して取得
    like_embeddings = engine.ids_to_embeds(like_ids)
    logger.info(f"like_embed.shape: {like_embeddings.shape}")
    logger.info(
        "like_embeddings l2norm: "
        f"{np.linalg.norm(like_embeddings, axis=1, ord=2)}"
    )
    return like_embeddings


def split_text(input_text):
    """
    検索クエリ文字列を空文字で分割して文字列のリストを返す。
    """
    return [x for x in input_text.replace("　", " ").strip().split(" ")]
