import logging
import sys
import time
from pprint import pprint

import gradio as gr
import numpy as np
import pandas as pd
import yaml

sys.path.append(".")
from src.models import Embedder, VectorEngine  # noqa: E402
from src.utils import get_device_info  # noqa: E402


def merge_embeddings(embeds1, embeds2):
    logger = logging.getLogger(__name__)
    logger.info(f"np.vstack({embeds1.shape}, {embeds2.shape})")
    embeds = np.vstack([embeds1, embeds2])
    result = np.mean(embeds, axis=0)
    return result


def split_text(input_text):
    return [x for x in input_text.replace("　", " ").strip().split(" ")]


def embedding_query(query, cast_int=False):
    global embedder
    logger = logging.getLogger(__name__)

    query_embeddings = np.zeros(engine.dimension)
    if query.strip():
        sentences = split_text(query)
        query_embeddings = embedder.encode(sentences)
        logger.info(f"query_embed.shape: {query_embeddings.shape}")
        logger.info(
            "query_embeddings l2norm: "
            f"{np.linalg.norm(query_embeddings, axis=1, ord=2)}"
        )

    return query_embeddings


def embedding_from_ids_string(like_ids):
    global engine
    logger = logging.getLogger(__name__)
    like_embeddings = np.zeros(engine.dimension)
    if like_ids.strip():
        like_ids = [int(x) for x in split_text(like_ids)]
        logger.info(f"found like_ids: {like_ids}")
        like_embeddings = engine.ids_to_embeddings(like_ids)
        logger.info(f"like_embed.shape: {like_embeddings.shape}")
        logger.info(
            "like_embeddings l2norm: "
            f"{np.linalg.norm(like_embeddings, axis=1, ord=2)}"
        )
    return like_embeddings


def format_to_text(df):
    """
    dataframe を text 形式にする。

    Parameters
    ------
    df: pandas.DataFrame
        入力 DataFrame。必須カラムは以下の通り。
        - id: int
          - 検索ID
        - similarity: float
          - 類似度
        - sentence: str
          - 埋め込み対象のテキスト

    Returns
    ------
    str
        表示用のテキスト
    """

    template = """## index: {index}
- info
  - id: {id}
  - similarity: {similarity}
  - url: {url}
  - length: {length}
- text

{sentence}

"""
    result = ""
    for index, row in df.iterrows():
        result += template.format(
            index=index,
            id=row["id"],
            similarity=row["similarity"],
            length=len(row["sentence"]),
            sentence=row["sentence"],
            url=row.get("url", ""),
        )

    return result


def search(query, like_ids, top_n):
    """
    検索クエリ、お気に入りID、推薦件数を受けとり結果を返す。

    Parameters
    ------
    query: str
        検索クエリ。空行で
    """
    global engine
    global text_df

    # init logger
    logger = logging.getLogger(__name__)
    logger.info(f"input: [{query}], [{like_ids}], [{top_n}]")

    # 検索クエリ文字列を埋め込みにする
    query_embeddings = embedding_query(query)

    # お気に入りIDを埋め込みにする
    like_embeddings = embedding_from_ids_string(like_ids)

    # ベクトル合成
    total_embedding = merge_embeddings(query_embeddings, like_embeddings)
    total_embedding_l2norm = np.linalg.norm(total_embedding, ord=2)
    logger.info(f"total_embedding l2norm: {total_embedding_l2norm}")

    # 合成ベクトルが 0 の場合
    if total_embedding_l2norm < 0.000001:
        return "検索できませんでした", None

    # 検索
    start_ts = time.perf_counter()
    similarities, ids = engine.search(total_embedding, top_n=top_n)
    elapsed_time = time.perf_counter() - start_ts
    logger.info(f"elapsed time: {elapsed_time}")

    # 結果を整形
    result_df = pd.DataFrame({"id": ids[0], "similarity": similarities[0]})
    result_df = pd.merge(result_df, text_df, on="id", how="left").loc[
        :, ["id", "similarity", "sentence", "url"]
    ]

    output_text = format_to_text(result_df)

    return str(elapsed_time), output_text


def main():
    global demo
    global embedder
    global engine
    global text_df
    global config

    # init logging
    logger = logging.getLogger(__name__)

    logger.info(pprint(get_device_info()))

    # load config
    config = yaml.safe_load(open("ui.yaml", "r"))["ui"]

    # load models
    logger.info("load models")
    embedder = Embedder(config["embedding_model"])  # noqa: F841
    engine = VectorEngine.load(config["vector_engine"])
    text_df = pd.read_parquet(config["sentences_data"])

    # logging
    logger.info(f"engine summary: {engine}")
    logger.info(f"embedder summary: {embedder}")

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
            top_n_number = gr.Number(value=config["default_top_n"])
            submit_button = gr.Button(value="検索")
            indicator_label = gr.Label(label="indicator")
            output_text = gr.Markdown(label="検索結果", show_label=True)

        # set event callback
        input_widgets = [query_text, like_ids, top_n_number]
        output_widgets = [indicator_label, output_text]
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
    main()
    demo.launch(share=config["gradio_share"], debug=True)
