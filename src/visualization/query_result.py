# import streamlit as st
import glob
import json
import re

import pandas as pd


def filter_model_name(model_string):
    result = ""
    for line in model_string.split("\n"):
        if "model_name_or_filepath" in line:
            result = re.sub("model_name_or_filepath': '", "", line)
    result = result.replace("'", "").replace(",", "").strip()
    return result


def main():
    runs = []
    for filepath in glob.glob(
        "data/processed/scenario/**/*.json", recursive=True
    ):
        run = json.load(open(filepath, "r"))
        runs.append(run)

    raw_df = pd.DataFrame(runs)
    df = pd.DataFrame()
    df["model_name"] = raw_df["configuration"].map(
        lambda x: filter_model_name(x["embedding_model"])
    )
    df["id"] = pd.DataFrame(raw_df["outputs"]).map(lambda x: x["result"]["id"])
    df["query"] = pd.DataFrame(raw_df["inputs"]).map(
        lambda x: x["positive_query"]
    )

    print(raw_df.columns)
    print(df.iloc[0]["model_name"])
    print(df)


if __name__ == "__main__":
    main()
