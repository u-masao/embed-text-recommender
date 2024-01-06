# import streamlit as st
import glob
import json

import pandas as pd


def main():
    runs = []
    for filepath in glob.glob(
        "data/processed/scenario/**/*.json", recursive=True
    ):
        run = json.load(open(filepath, "r"))
        runs.append(run)
        print(filepath)

    raw_df = pd.DataFrame(runs)
    df = pd.DataFrame()
    df["embedding_model"] = raw_df["configuration"].map(
        lambda x: json.loads(x["embedding_model"])
    )
    df["engine"] = raw_df["configuration"].map(
        lambda x: json.loads(x["engine"])
    )
    df["id"] = pd.DataFrame(raw_df["outputs"]).map(lambda x: x["result"]["id"])
    df["query"] = pd.DataFrame(raw_df["inputs"]).map(
        lambda x: x["positive_query"]
    )

    print(df.columns)
    rankings = []
    for index, item in df.iterrows():
        ranking_df = pd.DataFrame(
            [v for k, v in item["id"].items()], columns=["ranking"]
        )
        ranking_df["model_name"] = item["embedding_model"][
            "model_name_or_filepath"
        ]
        ranking_df["query"] = item["query"]
        rankings.append(ranking_df)

    ranking_df = pd.concat(rankings)
    ranking_df.to_csv("data/processed/ranking.csv", encoding="cp932")
    print(ranking_df)


if __name__ == "__main__":
    main()
