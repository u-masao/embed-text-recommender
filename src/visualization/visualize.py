import logging
import sys
import time

import cloudpickle
import gradio as gr
import numpy as np
import pandas as pd
import yaml

sys.path.append(".")
from src.features.build_features import VectorBuilder  # noqa: E402
from src.models.vector_engine import VectorEngine  # noqa: E402


def merge_embeddings(embeds1, embeds2):
    embeds = np.vstack(embeds1, embeds2)
    result = np.mean(embeds, axis=0)
    return result


def split_text(input_text):
    return [x for x in input_text.replace("　", " ").strip().split(" ")]


def search(query, like_ids, top_n):
    """
    検索クエリ、お気に入りID、推薦件数を受けとり結果を返す。

    Parameters
    ------
    query: str
        検索クエリ。空行で
    """
    global vector_builder
    global engine
    global text_df

    logger = logging.getLogger(__name__)

    logger.info(f"input: [{query}], [{like_ids}], [{top_n}]")

    # 検索クエリ文字列を埋め込みにする
    query_embeddings = None
    if query.strip():
        sentences = split_text(query)
        query_embeddings = vector_builder.encode(sentences)
        logger.info(f"embed.shape: {query_embeddings.shape}")
        logger.info(
            "query_embeddings l2norm: "
            f"{np.linalg.norm(query_embeddings, ord=2)}"
        )

    # お気に入りIDを埋め込みにする
    like_embeddings = None
    if like_ids.strip():
        like_ids = [int(x) for x in split_text(like_ids)]
        logger.info(f"found like_ids: {like_ids}")
        like_embeddings = engine.ids_to_embeddings(like_ids)
        logger.info(f"like_embed.shape: {like_embeddings.shape}")
        logger.info(
            "like_embeddings l2norm: "
            f"{np.linalg.norm(like_embeddings, ord=2)}"
        )

    # エラー処理
    if query_embeddings is None and like_embeddings is None:
        return "検索できませんでした", pd.DataFrame()

    # ベクトル合成
    total_embedding = merge_embeddings(query_embeddings, like_embeddings)
    logger.info(
        f"total_embedding l2norm: {np.linalg.norm(total_embedding, ord=2)}"
    )

    # 検索
    start_ts = time.perf_counter()
    similarities, ids = engine.search(total_embedding, top_n=top_n)
    elapsed_time = time.perf_counter() - start_ts
    logger.info(elapsed_time)

    # 結果を整形
    result_df = pd.DataFrame({"id": ids[0], "similarity": similarities[0]})
    result_df = pd.merge(result_df, text_df, on="id", how="left").loc[
        :, ["id", "similarity", "title", "url", "content", "category"]
    ]

    return str(elapsed_time), result_df


def main():
    global demo
    global vector_buider
    global engine
    global text_df

    # init logging
    logger = logging.getLogger(__name__)

    # load config
    config = yaml.safe_load(open("ui.yaml", "r"))["ui"]

    # load models
    logger.info("load models")
    vector_builder = VectorBuilder(config["embedding_model"])  # noqa: F841
    engine = VectorEngine.load(open(config["vector_engine"], "rb"))
    text_df, _ = cloudpickle.load(open(config["text_data"], "rb"))

    # make widgets
    with gr.Blocks() as demo:
        with gr.Column():
            query_text = gr.Textbox(
                label="検索クエリ", show_label=True, value=config["default_query"]
            )
            like_ids = gr.Textbox(
                label="お気に入り記事の id",
                show_label=True,
                value=config["default_like_ids"],
            )
            top_n_number = gr.Number(value=30)
            submit_button = gr.Button(value="検索")
            indicator_label = gr.Label(label="indicator")
            output_dataframe = gr.DataFrame(
                label="検索結果", show_label=True, interactive=True
            )

        # set event callback
        input_widgets = [query_text, like_ids, top_n_number]
        output_widgets = [indicator_label, output_dataframe]
        for entry_point in [
            query_text.submit,
            like_ids.submit,
            submit_button.click,
        ]:
            entry_point(
                fn=search,
                inputs=input_widgets,
                outputs=output_widgets,
            )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    demo.launch(share=False)
