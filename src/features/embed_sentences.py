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

    # make embedding_model
    embedding_model = EmbeddingModel.make_embedding_model(
        kwargs["embedding_model_string"],
        chunk_method=kwargs["chunk_method"],
        batch_size=kwargs["batch_size"],
    )

    # embedding
    embeddings = embedding_model.embed(df["sentence"])

    # output
    Path(kwargs["output_filepath"]).parent.mkdir(exist_ok=True, parents=True)
    with open(kwargs["output_filepath"], "wb") as fo:
        cloudpickle.dump(embeddings, fo)

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
@click.argument("output_filepath", type=click.Path())
@click.option(
    "--embedding_model_string",
    type=str,
    default="SentenceTransformer/oshizo/sbert-jsnli-luke-japanese-base-lite",
)
@click.option("--chunk_method", type=str, default="chunk_split")
@click.option("--batch_size", type=int, default=32)
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):
    # init logging
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiment("build_features")
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
