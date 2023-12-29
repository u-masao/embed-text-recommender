import pytest

from src.models import Embedder


@pytest.fixture(scope="session")
def embedder():
    # model_name = "intfloat/multilingual-e5-base"
    model_name = "intfloat/multilingual-e5-small"
    _embedder = Embedder(model_name)
    return _embedder


def test_bugfix_check_issue_17(embedder):
    """
    https://github.com/u-masao/sbert-news-recommender/issues/17
    """
    # 初期化されてるか
    assert embedder is not None, "Embedder が正しく初期化されてません"

    # 正常ケース
    scentences = ["猫と脳波以外のニュース", "犬"]
    embed = embedder.encode(scentences)
    assert embed is not None

    # 異常ケース
    # 空文字を入れると重み計算に失敗して Embedder 内の Assert に引っかかる
    scentences = ["猫と脳波以外のニュース", "", "犬"]
    embed = embedder.encode(scentences)
    assert embed is not None
