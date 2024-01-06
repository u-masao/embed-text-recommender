import glob
import json

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, linkage


@st.cache_data
def load_query_results():
    runs = []
    for filepath in glob.glob(
        "data/processed/scenario/**/*.json", recursive=True
    ):
        run = json.load(open(filepath, "r"))
        runs.append(run)

    return pd.DataFrame(runs)


def make_features(raw_df):
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

    rankings = []
    for index, item in df.iterrows():
        ranking_df = pd.DataFrame(
            [v for k, v in item["id"].items()], columns=["id"]
        )
        ranking_df["model_name"] = item["embedding_model"][
            "model_name_or_filepath"
        ]
        ranking_df["query"] = item["query"]
        ranking_df["rank"] = ranking_df.index + 1
        rankings.append(ranking_df)

    return pd.concat(rankings)


def _plot_dendrogram(temp_df, title=""):
    labels = temp_df.index
    linked = linkage(temp_df.map(lambda x: np.log10(x)), method="ward")

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))
    ax = ax.flatten()
    dendrogram_result = dendrogram(
        linked,
        orientation="right",
        distance_sort="descending",
        show_leaf_counts=True,
        ax=ax[0],
        labels=labels,
    )
    ax[0].set_title(title + "のデンドログラム")
    ax[0].set_ylabel("index")
    ax[0].set_ylabel("distance")

    sorted_df = temp_df.iloc[dendrogram_result["leaves"]]

    for index, cols in sorted_df.T.iterrows():
        ax[1].plot(cols.values, cols.index, label=cols.name)
    ax[1].set_title(title + "のランキング")

    st.pyplot(fig)

    st.dataframe(sorted_df)


def plot_ranking(
    df, y_axis="model_name", x_axis="rank", groupby="query", id_column="id"
):
    for cut_key in df[groupby].unique():
        title = f"「{cut_key}」"
        temp_df = df[df[groupby] == cut_key]
        temp_df = (
            temp_df.sort_values(id_column)
            .drop(groupby, axis=1)
            .set_index([y_axis, id_column])
            .unstack()[x_axis]
        )

        _plot_dendrogram(temp_df, title)


def main():
    raw_df = load_query_results()
    ranking_df = make_features(raw_df)

    plot_ranking(
        ranking_df, y_axis="model_name", x_axis="rank", groupby="query"
    )
    plot_ranking(
        ranking_df, y_axis="query", x_axis="rank", groupby="model_name"
    )


if __name__ == "__main__":
    main()
