import os
import threading
from pprint import pformat

import yaml


class ConfigurationManager:
    _instance = None
    _lock = threading.Lock()
    _config_prefix = "ui"
    _default_config_filepath = "ui.yaml.default"

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigurationManager, cls).__new__(cls)
                cls._instance.config = {}
        return cls._instance

    def load(self, filepath):
        if os.path.exists(filepath):
            self.config = yaml.safe_load(
                open(filepath, "r", encoding="utf-8")
            )[ConfigurationManager._config_prefix]
        else:
            if os.path.exists(self._default_config_filepath):
                self.load(self._default_config_filepath)
            else:
                raise ValueError(
                    "設定ファイルもデフォルトファイルもロードできません: "
                    f"{self._default_config_filepath}"
                )
        return self

    def save(self, filepath):
        yaml.dump(
            {ConfigurationManager._config_prefix: self.config},
            open(filepath, "w", encoding="utf-8"),
        )

        return self

    def __getitem__(self, key):
        return self.config.get(key, None)

    def __setitem__(self, key, value):
        self.config[key] = value

    def __str__(self):
        return pformat(self.config)
