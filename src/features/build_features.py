import logging
from pathlib import Path

import click
import cloudpickle
import mlflow
import pandas as pd

from src.models.embedding import EmbeddingModel
from src.utils import get_device_info


def embedding(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load dataset
    df = pd.read_parquet(kwargs["input_filepath"])

    # limit sentence size
    if kwargs["limit_sentence_size"] > 0:
        df = df.head(kwargs["limit_sentence_size"])

    # make features
    df["sentence"] = df["title"] + "\n" + df["content"]

    # make embedding_model
    embedding_model = EmbeddingModel.make_embedding_model(
        kwargs["embedding_storategy"],
        kwargs["model_name_or_filepath"],
        method=kwargs["chunking_method"],
    )

    # embedding
    embeddings = embedding_model.embed(df["sentence"])

    # output
    Path(kwargs["embeddings_filepath"]).parent.mkdir(
        exist_ok=True, parents=True
    )
    Path(kwargs["sentences_filepath"]).parent.mkdir(
        exist_ok=True, parents=True
    )
    with open(kwargs["embeddings_filepath"], "wb") as fo:
        cloudpickle.dump(embeddings, fo)
    df.to_parquet(kwargs["sentences_filepath"])

    # logging
    log_params = {
        "output.length": len(df),
        "output.columns": df.shape[1],
    }
    mlflow.log_params(log_params)
    logger.info(log_params)
    logger.info(f"embedding_model summary: {embedding_model}")
    logger.info(f"output dataframe: \n{df}")
    logger.info(f"output columns: \n{df.columns}")


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("sentences_filepath", type=click.Path())
@click.argument("embeddings_filepath", type=click.Path())
@click.option(
    "--model_name_or_filepath",
    type=str,
    default="oshizo/sbert-jsnli-luke-japanese-base-lite",
)
@click.option("--embedding_storategy", type=str, default="SentenceTransformer")
@click.option("--chunking_method", type=str, default="chunk_split")
@click.option("--limit_sentence_size", type=int, default=0)
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):
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
    embedding(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
