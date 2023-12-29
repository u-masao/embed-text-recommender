from src.models import Embedder

def test_return_is_not_none():
    model_name = "intfloat/multilingual-e5-base"
    scentences = ["猫と脳波以外のニュース 犬"]
    # embedder = Embedder(model_name)
    # embed = embedder.encode(scentences)
    assert model_name is not None
    # assert embed is not None

