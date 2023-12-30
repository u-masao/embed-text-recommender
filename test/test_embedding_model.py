import numpy as np
import pytest

from src.models.embedding import DummyEmbedding, EmbeddingModel


@pytest.fixture(scope="session")
def model():
    _model = EmbeddingModel(DummyEmbedding())
    yield _model


def test_basic(model):
    # 初期化チェック
    assert model is not None, "Embedding が正しく初期化されていません"

    # Embedding
    sentences = ["日本", "東京", "シンガポール", "ジャカルタ"]
    embed = model.embed(sentences)
    dimension = model.get_embed_dimension()

    # 返り値のチェック
    assert embed is not None, "embed() が値を返しません"
    assert embed.shape[0] == len(sentences)
    assert embed.shape[1] == dimension
    assert np.linalg.norm(embed[0], ord=2) > 0
