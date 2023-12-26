import logging
import sys
import time

import cloudpickle
import gradio as gr
import numpy as np
import pandas as pd

sys.path.append(".")
from src.features.build_features import VectorBuilder  # noqa: E402

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

engine = cloudpickle.load(open("data/interim/engine.cloudpickle", "rb"))
vector_builder = VectorBuilder("oshizo/sbert-jsnli-luke-japanese-base-lite")
text_df, embeddings = cloudpickle.load(
    open("data/interim/vectors.cloudpickle", "rb")
)

text_df["id"] = text_df["id"].astype("int64")


def search(query, like_ids, top_n):
    logger = logging.getLogger(__name__)

    logger.info(f"input: [{query}], [{like_ids}], [{top_n}]")

    query_embeddings = None
    like_embeddings = None
    if query.strip():
        sentences = [x.strip() for x in query.strip().split(" ")]
        query_embeddings = vector_builder.encode(sentences)
        logger.info(f"embed.shape: {query_embeddings.shape}")
        query_embedding = np.mean(query_embeddings, axis=0)
    else:
        query_embedding = np.zeros(engine.dimension)

    if like_ids.strip():
        like_ids = [int(x) for x in like_ids.strip().split(" ")]
        logger.info(f"found like_ids: {like_ids}")
        like_embeddings = engine.ids_to_embeddings(like_ids)
        logger.info(f"like_embed.shape: {like_embeddings.shape}")
        like_embedding = np.mean(like_embeddings, axis=0)
    else:
        like_embedding = np.zeros(engine.dimension)

    if query_embeddings is None and like_embeddings is None:
        return pd.DataFrame()

    total_embedding = query_embedding + like_embedding
    logger.info(
        f"total_embedding l2norm: {np.linalg.norm(total_embedding, ord=2)}"
    )

    start_ts = time.perf_counter()
    similarities, ids = engine.search(total_embedding, top_n=top_n)
    elapsed_time = time.perf_counter() - start_ts
    logger.info(elapsed_time)

    result_df = pd.DataFrame({"id": ids[0], "similarity": similarities[0]})
    result_df = pd.merge(result_df, text_df, on="id", how="left")
    return (
        str(elapsed_time),
        result_df[["id", "similarity", "title", "url", "content", "category"]],
    )


with gr.Blocks() as demo:
    with gr.Column():
        query_text = gr.Textbox(label="検索クエリ", show_label=True, value="東京")
        like_ids = gr.Textbox(
            label="お気に入り記事", show_label=True, value="6588884 6592773"
        )
        top_n_number = gr.Number(value=30)
        submit_button = gr.Button(value="検索")
        indicator_label = gr.Label(label="indicator")
        output_dataframe = gr.DataFrame(
            label="検索結果", show_label=True, interactive=True
        )
    query_text.submit(
        fn=search,
        inputs=[query_text, like_ids, top_n_number],
        outputs=[indicator_label, output_dataframe],
    )
    like_ids.submit(
        fn=search,
        inputs=[query_text, like_ids, top_n_number],
        outputs=[indicator_label, output_dataframe],
    )
    submit_button.click(
        fn=search,
        inputs=[query_text, like_ids, top_n_number],
        outputs=[indicator_label, output_dataframe],
    )


if __name__ == "__main__":
    demo.launch(share=False)
