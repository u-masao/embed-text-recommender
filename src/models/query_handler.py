import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

sys.path.append(".")
from src.models.embedding import EmbeddingModel  # noqa: E402
from src.models.search_engine import SearchEngine  # noqa: E402
from src.utils import ConfigurationManager  # noqa: E402
from src.utils import get_device_info  # noqa: E402
from src.utils import make_log_dict  # noqa: E402

CONFIG_FILEPATH = "ui.yaml"
MINIMUM_L2_NORM = 0.000001


class QueryHandler:
    def __init__(self, config_filepath=CONFIG_FILEPATH):
        self.config = ConfigurationManager().load(config_filepath)
        self.embedding_model, self.engine, self.text_df = self.init_models()

    def init_models(self, model_name=None):
        # init logging
        logger = logging.getLogger(__name__)
        logger.info(get_device_info())

        self.set_active_model(model_name)

        # get active model
        (
            embedding_model_string,
            search_engine_filepath,
        ) = self.get_active_model_name()

        # load embedding_model
        embedding_model = EmbeddingModel.make_embedding_model(
            embedding_model_string,
            chunk_method=self.config["chunk_method"],
        )

        # init search engin
        engine = SearchEngine.load(search_engine_filepath)

        # load text data
        text_df = pd.read_parquet(self.config["sentences_data"])

        # logging
        logger.info(f"engine summary: {engine}")
        logger.info(f"embedding_model summary: {embedding_model}")

        return embedding_model, engine, text_df

    def update_config(self, new_model_name):
        # reload config
        self.config = ConfigurationManager().load(CONFIG_FILEPATH)

        # cleanup
        del self.embedding_model, self.engine, self.text_df

        # init models
        self.embedding_model, self.engine, self.text_df = self.init_models(
            new_model_name
        )

        # save config
        self.config.save(CONFIG_FILEPATH)
        return self.config_to_string()

    def set_active_model(self, model_name=None):
        model_list = self.config["embedding_model_strings"]

        if len(model_list) == 0:
            raise ValueError("model_list is empty")

        if model_name is None:
            self.config["active_embedding_model_name"] = model_list[0]
            return

        if model_name not in model_list:
            raise ValueError(f"指定されたモデルはありません: {model_name}")

        self.config["active_embedding_model_name"] = model_name

    def get_active_model_name(self):
        embedding_model_string = self.config["active_embedding_model_name"]

        search_engine_filepath = (
            Path(self.config["models_directory"])
            / embedding_model_string
            / self.config["search_engine"]
        )

        return embedding_model_string, search_engine_filepath

    def mask_not_zero_vector(self, input_matrix):
        """
        return not zero vector, bool array.
        """
        return np.linalg.norm(input_matrix, axis=-1, ord=2) > 1e-10

    def merge_embeddings(
        self,
        positive_query_embeddings,
        positive_query_blend_ratio,
        negative_query_embeddings,
        negative_query_blend_ratio,
        like_embeddings,
        like_blend_ratio,
        dislike_embeddings,
        dislike_blend_ratio,
    ):
        """
        複数の埋め込みベクトルの平均を返す。
        """
        embeds_list = []

        for embeds, ratio in zip(
            [
                positive_query_embeddings,
                negative_query_embeddings,
                like_embeddings,
                dislike_embeddings,
            ],
            [
                positive_query_blend_ratio,
                -negative_query_blend_ratio,
                like_blend_ratio,
                -dislike_blend_ratio,
            ],
        ):
            # L2norm がゼロではないベクトルだけを登録する
            mask = self.mask_not_zero_vector(embeds)
            if mask.sum() > 0:
                if embeds[mask].flatten().shape[0] > 0:
                    embeds_list.append(
                        embeds[mask] * ratio,
                    )

        # calc mean
        return np.mean(np.vstack(embeds_list), axis=0)

    def split_text(self, input_text):
        """
        検索クエリ文字列を空文字で分割して文字列のリストを返す。
        """
        return [x for x in input_text.replace("　", " ").strip().split(" ")]

    def embedding_query(self, query):
        """
        検索クエリ文字列を埋め込みベクトルに変換する。
        """

        logger = logging.getLogger(__name__)

        # 結果変数を初期化
        query_embeddings = np.zeros(self.embedding_model.get_embed_dimension())

        # 入力文字列が空の場合はゼロベクトルを返す
        if len(query.strip()) == 0:
            return query_embeddings

        # 検索クエリをベクトル化
        sentences = self.split_text(query)
        query_embeddings = self.embedding_model.embed(sentences)
        logger.info(f"query_embed.shape: {query_embeddings.shape}")
        logger.info(
            "query_embeddings l2norm: "
            f"{np.linalg.norm(query_embeddings, axis=1, ord=2)}"
        )

        return query_embeddings

    def embedding_from_ids_string(self, like_ids):
        """
        id 一覧文字列をベクトルに変換する。
        """

        # Logger を初期化
        logger = logging.getLogger(__name__)

        # 結果のベクトルを初期化
        like_embeddings = np.zeros(self.engine.get_embed_dimension())

        # ID 一覧が空の場合は ゼロベクトルを返す
        if len(like_ids.strip()) == 0:
            return like_embeddings

        # int のリストに変換 (変換に失敗すると ValueError が出る)
        like_ids = [int(x) for x in self.split_text(like_ids)]
        logger.info(f"found like_ids: {like_ids}")

        # 埋め込みを検索して取得
        like_embeddings = self.engine.ids_to_embeds(like_ids)
        logger.info(f"like_embed.shape: {like_embeddings.shape}")
        logger.info(
            "like_embeddings l2norm: "
            f"{np.linalg.norm(like_embeddings, axis=1, ord=2)}"
        )
        return like_embeddings

    def search(
        self,
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

        # init logger
        logger = logging.getLogger(__name__)
        logger.info(
            f"input: [{positive_query}], [{negative_query}],"
            f" [{like_ids}], [{top_n}]"
        )

        # 検索クエリ文字列を埋め込みにする
        start_ts = time.perf_counter()
        positive_query_embeddings = self.embedding_query(positive_query)
        negative_query_embeddings = self.embedding_query(negative_query)
        encode_elapsed_time = time.perf_counter() - start_ts

        # お気に入りIDを埋め込みにする
        like_embeddings = self.embedding_from_ids_string(like_ids)
        dislike_embeddings = self.embedding_from_ids_string(dislike_ids)

        # ベクトル合成
        total_embedding = self.merge_embeddings(
            positive_query_embeddings,
            positive_query_blend_ratio,
            negative_query_embeddings,
            negative_query_blend_ratio,
            like_embeddings,
            like_blend_ratio,
            dislike_embeddings,
            dislike_blend_ratio,
        )

        # L2ノルム計算(長さ計算)
        total_embedding_l2norm = np.linalg.norm(total_embedding, ord=2)
        logger.info(f"total_embedding l2norm: {total_embedding_l2norm}")

        # 合成ベクトルが 0 の場合
        if total_embedding_l2norm < MINIMUM_L2_NORM:
            return "検索できません。検索キーのベクトルの長さが 0 になってしまいました。", None

        # 検索
        start_ts = time.perf_counter()
        ids, similarities = self.engine.search(total_embedding, top_n=top_n)
        search_elapsed_time = time.perf_counter() - start_ts

        # 結果を整形
        start_ts = time.perf_counter()
        result_df = pd.DataFrame({"id": ids[0], "similarity": similarities[0]})
        result_df = (
            pd.merge(result_df, self.text_df, on="id", how="left")
            .loc[:, ["id", "similarity", "sentence", "url"]]
            .fillna("")
        )  # sentence, url が無い場合に作動
        df_merge_elapsed_time = time.perf_counter() - start_ts

        # log result
        self.log_search_result(
            {
                "inputs": {
                    "positive_query": positive_query,
                    "positive_query_blend_ratio": positive_query_blend_ratio,
                    "negative_query": negative_query,
                    "negative_query_blend_ratio": negative_query_blend_ratio,
                    "like_ids": like_ids,
                    "like_blend_ratio": like_blend_ratio,
                    "dislike_ids": dislike_ids,
                    "dislike_blend_ratio": dislike_blend_ratio,
                    "top_n": top_n,
                },
                "embeds": {
                    "positive_query_embeddings": positive_query_embeddings,
                    "negative_query_embeddings": negative_query_embeddings,
                    "like_embeddings": like_embeddings,
                    "dislike_embeddings": dislike_embeddings,
                    "total_embedding": total_embedding,
                },
                "outputs": {"result": result_df},
                "elapsed_time": {
                    "embeddindg": encode_elapsed_time,
                    "search": search_elapsed_time,
                    "df_merge": df_merge_elapsed_time,
                },
                "meta": {
                    "timestamp": time.time(),
                },
                "configuration": {
                    "embedding_model": str(self.embedding_model),
                    "engine": str(self.engine),
                },
            }
        )

        # 動作状況メッセージを作成
        message = (
            "```"
            f"encode: {encode_elapsed_time:.3f} sec"
            f"\nsearch: {search_elapsed_time:.3f} sec"
            f"\ntext merge: {df_merge_elapsed_time:.3f} sec"
            f"\nmodel: {str(self.embedding_model)}"
            f"\nengine: {str(self.engine)}"
            f"\nmodel dimension: {self.embedding_model.get_embed_dimension()}"
            "```"
        )
        output_text = self.format_to_text(result_df)
        return message, output_text

    def format_to_text(self, df):
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

        template = """
*******

## result: {index}

- info
  - id: {id}
  - similarity: {similarity}
  - url: {url}
  - length: {length}

{sentence}

    """
        result = ""
        for index, row in df.iterrows():
            result += template.format(
                index=index + 1,
                id=row["id"],
                similarity=row["similarity"],
                length=len(row["sentence"]),
                sentence=row["sentence"],
                url=row.get("url", ""),
            )

        return result

    def log_search_result(self, result):
        """
        検索結果をファイルに保存する。
        """
        # make dirs
        log_dir = Path(self.config["log_dir"])
        log_dir.mkdir(exist_ok=True, parents=True)

        # バイナリで上書き保存
        cloudpickle.dump(
            result,
            open(log_dir / "last_search_result_detail.cloudpickle", "wb"),
        )

        # JSON で 保存
        ts = datetime.now()
        filename = ts.strftime("search_result_%Y%m%d_%H%M%S_%f.json")
        json.dump(
            make_log_dict(result), open(log_dir / filename, "w"), indent=2
        )

    def config_to_string(self):
        """
        設定情報を文字列表現にする。
        Markdown Widgetで表示することを想定。
        """
        return f"```\n{self.config}\n```"


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    query_handler = QueryHandler()
    result = query_handler.search("ダイビング", 1.0, "", 1.0, "", 1.0, "", 1.0, 30)
    logger.info(result)
