import os

from src.utils import ConfigurationManager


def test_basic():
    # values
    filepath = ".config.tmp.yaml"
    new_value = "value"
    key = "new_setting"

    # 初期化チェック
    config = ConfigurationManager()
    assert config

    # save
    config.save(filepath)

    # load
    another_config = ConfigurationManager()
    another_config.load(filepath)
    assert config[key] is None
    assert os.path.isfile(filepath)

    # 代入
    config[key] = new_value
    config.save(filepath)

    # load
    another_config = ConfigurationManager()
    another_config.load(filepath)
    assert config[key] == another_config[key]

    # remove file
    os.remove(filepath)
    assert os.path.isfile(filepath) is False

    # load default config
    config.load(filepath)
    assert config["models_directory"] is not None
