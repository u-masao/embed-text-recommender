import logging
import sys

import gradio as gr

sys.path.append(".")
from src.models import QueryHandler  # noqa: E402

demo = None  # for suppress gradio reload error


def init_widgets():
    """
    Widget を配置しイベントリスナーを設定する。
    """
    global query_handler

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
        gr.Markdown(value=get_ui_header())
        with gr.Column():
            # 入力 Widget
            with gr.Row():
                positive_query_text = gr.Textbox(
                    label="ポジティブ検索クエリ",
                    show_label=True,
                    scale=left_column_scale,
                    autofocus=True,
                )
                submit_button = gr.Button(value="検索")
            with gr.Accordion(label="複雑な条件設定", open=False):
                with gr.Row():
                    _ = gr.Label(
                        "",
                        scale=left_column_scale,
                    )
                    positive_blend_ratio = gr.Slider(
                        label="pos. ブレンド倍率",
                        **slider_kwargs,
                    )
                with gr.Row():
                    negative_query_text = gr.Textbox(
                        label="ネガティブ検索クエリ",
                        show_label=True,
                        scale=left_column_scale,
                    )
                    negative_blend_ratio = gr.Slider(
                        label="neg. ブレンド倍率",
                        **slider_kwargs,
                    )
                with gr.Row():
                    like_ids = gr.Textbox(
                        label="お気に入り記事の id",
                        show_label=True,
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
                        scale=left_column_scale,
                    )
                    dislike_ids_blend_ratio = gr.Slider(
                        label="dislike ブレンド倍率",
                        **slider_kwargs,
                    )
                top_n_number = gr.Number(
                    value=query_handler.config["default_top_n"],
                    label="取得件数",
                    show_label=True,
                )
                set_example_button = gr.Button(value="サンプル値をセット")
                clear_button = gr.Button(value="クリア")

            # 設定情報
            with gr.Accordion(label="設定", open=False):
                active_model_string, _ = query_handler.get_active_model_name()
                model_selector = gr.Dropdown(
                    choices=query_handler.config["embedding_model_strings"],
                    value=active_model_string,
                    type="index",
                    label="モデル選択",
                    show_label=True,
                )
                config_markdown = gr.Markdown(
                    query_handler.config_to_string(), label="現在の設定情報"
                )

            # オリジナルテキスト表示
            display_original_text()

            # 出力 Widget
            with gr.Accordion(label="モデルの詳細情報", open=False):
                indicator_markdown = gr.Markdown(
                    label="indicator",
                    show_label=True,
                    value="",
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
                fn=query_handler.search,
                inputs=input_widgets,
                outputs=output_widgets,
            )
        model_selector.change(
            fn=query_handler.update_config,
            inputs=[model_selector],
            outputs=[config_markdown],
        )
        clear_button.click(fn=clear_widgets, outputs=input_widgets)
        set_example_button.click(fn=set_example_widgets, outputs=input_widgets)
    return demo


def clear_widgets():
    return (
        "",  # positive_query
        1.0,  # positive_query_blend_ratio
        "",  # negative_query
        1.0,  # negative_query_blend_ratio
        "",  # like_ids
        1.0,  # like_blend_ratio
        "",  # dislike_ids
        1.0,  # dislike_blend_ratio
        query_handler.config["default_top_n"],  # top_n
    )


def get_ui_header():
    return """
# Embed Text Recommender

様々な埋め込みモデルを利用して文書を検索することができます。
    """


def set_example_widgets():
    return (
        query_handler.config["default_positive_query"],  # positive_query
        1.0,  # positive_query_blend_ratio
        query_handler.config["default_negative_query"],  # negative_query
        1.0,  # negative_query_blend_ratio
        query_handler.config["default_like_ids"],  # like_ids
        1.0,  # like_blend_ratio
        query_handler.config["default_dislike_ids"],  # dislike_ids
        1.0,  # dislike_blend_ratio
        query_handler.config["default_top_n"],  # top_n
    )


def display_original_text(
    sample_size=1000,
    text_length=200,
):
    # 表示対象の DataFrame を変数に保存
    target_df = query_handler.text_df

    # サンプリングと整形
    df_size = target_df.shape[0]
    sample_size = min(sample_size, df_size)
    sampled_df = (
        target_df.loc[:, ["id", "sentence"]]
        .sample(sample_size)
        .sort_values("id")
    )
    sampled_df["sentence_length"] = sampled_df["sentence"].str.len()
    sampled_df["sentence"] = sampled_df["sentence"].str[:text_length]

    # widget 表示
    with gr.Accordion(label="オリジナルテキスト", open=False):
        gr.Dataframe(
            sampled_df,
            label=f"{len(sampled_df)} / {df_size} 件をサンプリングしています",
            show_label=True,
            interactive=True,
        )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    query_handler = QueryHandler()
    demo = init_widgets()
    demo.launch(share=query_handler.config["gradio_share"], debug=True)
