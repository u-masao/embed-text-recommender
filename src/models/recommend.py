import logging

import click
import mlflow

from src.models.embedding import EmbeddingModel
from src.models.search_engine import SearchEngine
from src.utils import get_device_info


def recommend(kwargs):
    # init logger
    logger = logging.getLogger(__name__)

    # make embedding_model
    embedding_model = EmbeddingModel.make_embedding_model(
        kwargs["embedding_storategy"],
        kwargs["model_name_or_filepath"],
        method=kwargs["chunk_method"],
    )

    # load search engine
    engine = SearchEngine.load(kwargs["search_engine_filepath"])
    logger.info(f"engine summary: {engine}")

    for sentences, like_ids in zip([["飼い犬"]], [[6588884, 6592773]]):
        embeddings = embedding_model.embed(sentences)
        similar_ids, similarities = engine.search(embeddings)
        logger.info(embeddings.shape)
        logger.info(similarities)
        logger.info(similar_ids)

        like_embeddings = engine.ids_to_embeds(like_ids)
        logger.info(like_ids)
        logger.info(like_embeddings.shape)


@click.command()
@click.argument("search_engine_filepath", type=click.Path(exists=True))
@click.option(
    "--model_name_or_filepath",
    type=str,
    default="oshizo/sbert-jsnli-luke-japanese-base-lite",
)
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--embedding_storategy", type=str, default="SentenceTransformer")
def main(**kwargs):
    """
    クエリ文字列に類似するテキストを計算する。

    Parameters
    ------
    kwargs: Dict[str, any]
        CLI オプション
        - search_engine_filepath
            ベクトルエンジンバイナリのファイルパス
        - model_name_or_filepath
            埋め込み作成モデル名
        - mlflow_run_name
            MLflow の run_name
    """
    # init logging
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiment("recommend")
    mlflow.start_run(run_name="mlflow_run_name")

    # log cli options
    logger.info(f"args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})
    mlflow.log_params(get_device_info())

    # process
    recommend(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
