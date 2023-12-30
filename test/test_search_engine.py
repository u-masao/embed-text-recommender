import numpy as np
import pytest

from src.models.search_engine import (
    DummySearchEngine,
    FaissSearchEngine,
    SearchEngine,
)

# 埋め込み表現の次元数
EMBEDDING_DIMENSION = 10


@pytest.fixture(
    scope="session",
    params=[
        DummySearchEngine(dimension=EMBEDDING_DIMENSION),
        FaissSearchEngine(dimension=EMBEDDING_DIMENSION),
    ],
)
def engine(request):
    """
    SearchEngine を初期化してテストに渡す
    """
    return SearchEngine(request.param)


@pytest.fixture(
    scope="session",
    params=[
        ([1], np.random.randn(1, EMBEDDING_DIMENSION)),
        ([2], np.random.randn(EMBEDDING_DIMENSION)),
        ([1, 2, 3, 4], np.random.randn(4, EMBEDDING_DIMENSION)),
    ],
)
def ids_embeds(request):
    """
    ID と 埋め込み表現を初期化してテストに渡す
    """
    return request.param


@pytest.fixture(
    scope="session",
    params=[
        # 0,
        1,
        10,
        20,
    ],
)
def top_n(request):
    """
    top_n を初期化してテストに渡す
    """
    return request.param


def test_basic(engine, ids_embeds, top_n):
    # 初期化チェック
    assert engine is not None, "SearchEngine が正しく初期化されていません"

    # 変数を定義
    ids, embeds = ids_embeds

    # use engine
    dimension = engine.get_embed_dimension()
    engine.add_ids_and_embeds(ids, embeds)
    result_ids, similarities = engine.search(embeds, top_n=top_n)

    # 返り値のチェック
    assert result_ids is not None
    assert similarities is not None
    assert dimension > 0
    assert embeds.ndim == 1 or embeds.ndim == 2
    assert result_ids.shape[-1] == top_n
    assert similarities.shape[-1] == top_n

    # 入力の埋め込みの形によって場合分け
    if embeds.ndim == 1:
        assert result_ids.shape[0] == 1
        assert similarities.shape[0] == 1
    elif embeds.ndim == 2:
        assert result_ids.shape[0] == embeds.shape[0]
        assert similarities.shape[0] == embeds.shape[0]
