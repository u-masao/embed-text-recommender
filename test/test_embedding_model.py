import numpy as np
import pytest

from src.models.embedding import (
    DummyEmbedding,
    EmbeddingModel,
    SentenceTransformerEmbedding,
    Word2VecEmbedding,
)


@pytest.fixture(
    scope="session",
    params=[
        DummyEmbedding(),
        SentenceTransformerEmbedding(
            "oshizo/sbert-jsnli-luke-japanese-base-lite"
        ),
        Word2VecEmbedding("models/Word2Vec/base_dict/kv.bin"),
    ],
)
def model(request):
    return EmbeddingModel(request.param)


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


def test_sentence_transformer_embedding():
    sentences = ["日本" * 500, "東京" * 500, "シンガポール" * 500, "ジャカルタ" * 500]
    for chunk_method in ["chunk_split", "head_only"]:
        # init model
        model = SentenceTransformerEmbedding(
            "oshizo/sbert-jsnli-luke-japanese-base-lite",
            chunk_method=chunk_method,
        )

        # 初期化チェック
        assert model is not None, "Embedding が正しく初期化されていません"
        # Embedding
        embed = model.embed(sentences)
        dimension = model.get_embed_dimension()

        # 返り値のチェック
        assert embed is not None, "embed() が値を返しません"
        assert embed.shape[0] == len(sentences)
        assert embed.shape[1] == dimension
        assert np.linalg.norm(embed[0], ord=2) > 0
