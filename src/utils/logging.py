from pprint import pprint

import numpy as np
import pandas as pd


def make_log_dict(data):
    if isinstance(data, np.ndarray):
        return {
            "shape": data.shape,
            "l2norm": np.linalg.norm(data, axis=-1, ord=2).tolist(),
        }

    if isinstance(data, pd.DataFrame):
        return data.to_dict()

    if isinstance(data, dict) is False:
        return data

    result = {}
    for key, value in data.items():
        result[key] = make_log_dict(value)
    return result


if __name__ == "__main__":
    import cloudpickle

    data = cloudpickle.load(
        open("data/processed/log/last_search_result.cloudpickle", "rb")
    )
    pprint(make_log_dict(data))
