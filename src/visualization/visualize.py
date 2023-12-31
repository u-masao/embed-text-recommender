import logging
import sys
import time
from pprint import pformat, pprint

import gradio as gr
import numpy as np
import pandas as pd
import yaml

sys.path.append(".")
from src.models.embedding import EmbeddingModel  # noqa: E402
from src.models.search_engine import SearchEngine  # noqa: E402
from src.utils import get_device_info  # noqa: E402

CONFIG_FILEPATH = "ui.yaml"
demo = None  # for suppress gradio reload error


def merge_embeddings(embeds_list):
    """
    複数の埋め込みベクトルの平均を返す。
    """
    embeds = np.vstack(embeds_list)
    result = np.mean(embeds, axis=0)
    return result


def split_text(input_text):
    """
    検索クエリ文字列を空文字で分割して文字列のリストを返す。
    """
    return [x for x in input_text.replace("　", " ").strip().split(" ")]


def embedding_query(query):
    """
    検索クエリ文字列を埋め込みベクトルに変換する。
    """

    global embedding_model
    logger = logging.getLogger(__name__)

    # 結果変数を初期化
    query_embeddings = np.zeros(engine.get_embed_dimension())

    # 入力文字列が空の場合はゼロベクトルを返す
    if len(query.strip()) == 0:
        return query_embeddings

    # 検索クエリをベクトル化
    sentences = split_text(query)
    query_embeddings = embedding_model.embed(sentences)
    logger.info(f"query_embed.shape: {query_embeddings.shape}")
    logger.info(
        "query_embeddings l2norm: "
        f"{np.linalg.norm(query_embeddings, axis=1, ord=2)}"
    )

    return query_embeddings


def embedding_from_ids_string(like_ids):
    """
    id 一覧文字列をベクトルに変換する。
    """
    global engine

    # Logger を初期化
    logger = logging.getLogger(__name__)

    # 結果のベクトルを初期化
    like_embeddings = np.zeros(engine.get_embed_dimension())

    # ID 一覧が空の場合は ゼロベクトルを返す
    if len(like_ids.strip()) == 0:
        return like_embeddings

    # int のリストに変換 (変換に失敗すると ValueError が出る)
    like_ids = [int(x) for x in split_text(like_ids)]
    logger.info(f"found like_ids: {like_ids}")

    # 埋め込みを検索して取得
    like_embeddings = engine.ids_to_embeds(like_ids)
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


def clear_widgets():
    global config

    return (
        "",  # positive_query
        1.0,  # positive_query_blend_ratio
        "",  # negative_query
        1.0,  # negative_query_blend_ratio
        "",  # like_ids
        1.0,  # like_blend_ratio
        "",  # dislike_ids
        1.0,  # dislike_blend_ratio
        config["default_top_n"],  # top_n
    )


def set_example_widgets():
    global config

    return (
        config["default_positive_query"],  # positive_query
        1.0,  # positive_query_blend_ratio
        config["default_negative_query"],  # negative_query
        1.0,  # negative_query_blend_ratio
        config["default_like_ids"],  # like_ids
        1.0,  # like_blend_ratio
        config["default_dislike_ids"],  # dislike_ids
        1.0,  # dislike_blend_ratio
        config["default_top_n"],  # top_n
    )


def search(
    positive_query,
    positive_query_blend_ratio,
    negative_query,
    negative_query_blend_ratio,
    like_ids,
    like_blend_ratio,
    dislike_ids,
    dislike_blend_ratio,
    top_n,
):
    """
    検索クエリ、お気に入りID、推薦件数を受けとり結果を返す。

    Parameters
    ------
    positive_query: str
        検索クエリ。
    positive_query_blend_ratio: float
        検索クエリベクトルのブレンド倍率。
    negative_query: str
        ネガティブ検索クエリ。
    negative_query_blend_ratio: float
        ネガティブ検索クエリベクトルのブレンド倍率。
    like_ids: str
        お気に入り ID
    like_blend_ratio: float
        お気に入りベクトルのブレンド倍率。
    dislike_ids: str
        見たくない ID
    dislike_blend_ratio: float
        見たくないベクトルのブレンド倍率。
    top_n: int
        取得したい件数

    Returns
    ------
    str
        メッセージ
    str
        検索結果のテキスト
    """
    global embedding_model
    global engine
    global text_df

    # init logger
    logger = logging.getLogger(__name__)
    logger.info(
        f"input: [{positive_query}], [{negative_query}],"
        f" [{like_ids}], [{top_n}]"
    )

    # 検索クエリ文字列を埋め込みにする
    start_ts = time.perf_counter()
    positive_query_embeddings = embedding_query(positive_query)
    negative_query_embeddings = embedding_query(negative_query)
    encode_elapsed_time = time.perf_counter() - start_ts

    # お気に入りIDを埋め込みにする
    like_embeddings = embedding_from_ids_string(like_ids)
    dislike_embeddings = embedding_from_ids_string(dislike_ids)

    # ベクトル合成
    total_embedding = merge_embeddings(
        [
            positive_query_embeddings * positive_query_blend_ratio,
            -negative_query_embeddings * negative_query_blend_ratio,
            like_embeddings * like_blend_ratio,
            -dislike_embeddings * dislike_blend_ratio,
        ]
    )

    # L2ノルム計算(長さ計算)
    total_embedding_l2norm = np.linalg.norm(total_embedding, ord=2)
    logger.info(f"total_embedding l2norm: {total_embedding_l2norm}")

    # 合成ベクトルが 0 の場合
    if total_embedding_l2norm < 0.000001:
        return "検索できません。検索キーのベクトルの長さが 0 になってしまいました。", None

    # 検索
    start_ts = time.perf_counter()
    ids, similarities = engine.search(total_embedding, top_n=top_n)
    search_elapsed_time = time.perf_counter() - start_ts

    # 結果を整形
    start_ts = time.perf_counter()
    result_df = pd.DataFrame({"id": ids[0], "similarity": similarities[0]})
    result_df = pd.merge(result_df, text_df, on="id", how="left").loc[
        :, ["id", "similarity", "sentence", "url"]
    ]
    df_merge_elapsed_time = time.perf_counter() - start_ts
    output_text = format_to_text(result_df)

    # 動作状況メッセージを作成
    message = (
        f"encode: {encode_elapsed_time:.3f} sec"
        f",\nsearch: {search_elapsed_time:.3f} sec"
        f",\ndf merge: {df_merge_elapsed_time:.3f} sec"
        f",\nmodel: {config['embedding_model_name']}"
        f",\nmodel dimension: {embedding_model.get_embed_dimension()}"
    )
    return message, output_text


def config_to_string(config):
    """
    設定情報を文字列表現にする。
    Markdown Widgetで表示することを想定。
    """
    return f"```\n{pformat(config)}'''"


def reload():
    global config
    global embedding_model
    global engine
    global text_df

    config = load_config()
    embedding_model, engine, text_df = init_models(config)
    return config_to_string(config)


def load_config():
    return yaml.safe_load(open(CONFIG_FILEPATH, "r"))["ui"]


def init_models(config):
    # init logging
    logger = logging.getLogger(__name__)
    logger.info(pprint(get_device_info()))

    # load embedding_model
    embedding_model = EmbeddingModel.make_embedding_model(
        config["embedding_storategy"],
        config["embedding_model_name"],
        chunk_method=config["chunk_method"],
    )  # noqa: F841

    # init search engin
    engine = SearchEngine.load(config["search_engine"])

    # load text data
    text_df = pd.read_parquet(config["sentences_data"])

    # logging
    logger.info(f"engine summary: {engine}")
    logger.info(f"embedding_model summary: {embedding_model}")

    return embedding_model, engine, text_df


def init_widgets(config):
    """
    Widget を配置しイベントリスナーを設定する。
    """

    slider_kwargs = dict(
        minimum=0.0,
        maximum=2.0,
        value=1.0,
        step=0.1,
        show_label=True,
        scale=1,
    )
    left_column_scale = 2

    with gr.Blocks() as demo:
        with gr.Column():
            # 入力 Widget
            with gr.Row():
                positive_query_text = gr.Textbox(
                    label="ポジティブ検索クエリ",
                    show_label=True,
                    # value=config["default_positive_query"],
                    scale=left_column_scale,
                )
                positive_blend_ratio = gr.Slider(
                    label="pos 検索クエリ ブレンド倍率",
                    **slider_kwargs,
                )
            with gr.Accordion(label="詳細な検索条件", open=False):
                with gr.Row():
                    negative_query_text = gr.Textbox(
                        label="ネガティブ検索クエリ",
                        show_label=True,
                        # value=config["default_negative_query"],
                        scale=left_column_scale,
                    )
                    negative_blend_ratio = gr.Slider(
                        label="neg 検索クエリ ブレンド倍率",
                        **slider_kwargs,
                    )
                with gr.Row():
                    like_ids = gr.Textbox(
                        label="お気に入り記事の id",
                        show_label=True,
                        # value=config["default_like_ids"],
                        scale=left_column_scale,
                    )
                    like_ids_blend_ratio = gr.Slider(
                        label="like ブレンド倍率",
                        **slider_kwargs,
                    )
                with gr.Row():
                    dislike_ids = gr.Textbox(
                        label="見たくない記事の id",
                        show_label=True,
                        # value=config["default_dislike_ids"],
                        scale=left_column_scale,
                    )
                    dislike_ids_blend_ratio = gr.Slider(
                        label="dislike ブレンド倍率",
                        **slider_kwargs,
                    )
                top_n_number = gr.Number(value=config["default_top_n"])
                set_example_button = gr.Button(value="サンプル値をセット")
                clear_button = gr.Button(value="クリア")
            submit_button = gr.Button(value="検索")

            # 設定情報
            with gr.Accordion(label="設定", open=False):
                config_markdown = gr.Markdown(config_to_string(config))
                initialize_button = gr.Button(value="設定ファイルのリロード")

            # 出力 Widget
            indicator_markdown = gr.Markdown(
                label="indicator",
                show_label=True,
                value=(
                    f"model: {config['embedding_model_name']}, "
                    f"model dimension: {embedding_model.get_embed_dimension()}"
                ),
            )
            output_text = gr.Markdown(label="検索結果", show_label=True)

        # set event callback
        input_widgets = [
            positive_query_text,
            positive_blend_ratio,
            negative_query_text,
            negative_blend_ratio,
            like_ids,
            like_ids_blend_ratio,
            dislike_ids,
            dislike_ids_blend_ratio,
            top_n_number,
        ]
        output_widgets = [indicator_markdown, output_text]
        for entry_point in [
            positive_query_text.submit,
            negative_query_text.submit,
            like_ids.submit,
            dislike_ids.submit,
            submit_button.click,
            top_n_number.submit,
        ]:
            entry_point(
                fn=search,
                inputs=input_widgets,
                outputs=output_widgets,
            )
        initialize_button.click(fn=reload, outputs=[config_markdown])
        clear_button.click(fn=clear_widgets, outputs=input_widgets)
        set_example_button.click(fn=set_example_widgets, outputs=input_widgets)
    return demo


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    config = load_config()
    embedding_model, engine, text_df = init_models(config)
    demo = init_widgets(config)
    demo.launch(share=config["gradio_share"], debug=True)
