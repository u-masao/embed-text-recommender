import numpy as np
import pandas as pd
import pytest

from src.models.search_engine import FaissSearchEngine, SearchEngine

# 埋め込み表現の次元数
EMBEDDING_DIMENSION = 30


@pytest.fixture(
    scope="session",
    params=[
        200,1000,
    ],
)
def id_size(request):
    """
    SearchEngine に登録する ID 数を返す
    """
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        100,
        20,
    ],
)
def top_n(request):
    """
    top_n を初期化してテストに渡す
    """
    return request.param


def _calc_cosine(db_embeds, query_embed):
    """
    コサインを計算（正規化&内積）
    """
    norm_db = (db_embeds.T / np.linalg.norm(db_embeds, axis=1, ord=2)).T
    norm_query = query_embed / np.linalg.norm(query_embed, ord=2)
    cosine = np.dot(norm_db, norm_query)
    return cosine


def _pickup_top_n(cosine, ids, top_n):
    """
    ソートして上位を返す
    """
    return (
        pd.DataFrame({"cosine_np": cosine, "id": ids})
        .sort_values("cosine_np", ascending=False)
        .head(top_n)
    )


def test_cosine(id_size, top_n):


    # SearchEngine への入力を作成
    input_ids= [x for x in range(id_size)]
    input_embeds =  np.random.randn(id_size, EMBEDDING_DIMENSION)

    # クエリーを作成
    offset = 0.1
    query_embed = input_embeds[0]+offset

    # use engine
    engine = SearchEngine(
        FaissSearchEngine(dimension=EMBEDDING_DIMENSION)
    )
    engine.add_ids_and_embeds(input_ids, input_embeds)
    result_ids, similarities = engine.search(query_embed, top_n=top_n)
    assert result_ids is not None
    assert similarities is not None

    # 結果を Pandas DateFrame にして結合
    search_engine_result_df = pd.DataFrame(
        {"id": result_ids[0], "cosine_se": similarities[0]}
    )
    numpy_result_df = _pickup_top_n(
        _calc_cosine(input_embeds, query_embed), input_ids, top_n
    )
    merged_df = pd.merge(
        search_engine_result_df, numpy_result_df, on="id", how="outer"
    )
    assert merged_df.shape[0] == top_n

    # 一致しない行があれば fillna(0) で誤差が大きくなる
    merged_df = merged_df.fillna(0)

    # SearchEngine の Numpy の計算結果し、二乗和が十分に小さいか？
    assert (
        (merged_df["cosine_se"] - merged_df["cosine_np"]) ** 2
    ).sum() < 1.0e-10
