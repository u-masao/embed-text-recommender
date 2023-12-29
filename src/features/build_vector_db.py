import logging

import click
import cloudpickle
import mlflow
import pandas as pd

from src.models import VectorEngine
from src.utils import get_device_info


def build_vector_db(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load sentences
    logger.info("load sentences")
    df = pd.read_parquet(kwargs["sentences_filepath"])

    # load embeddings
    logger.info("load embeddings file")
    with open(kwargs["embeddings_filepath"], "rb") as fo:
        embeddings = cloudpickle.load(fo)

    # build vector engine
    logger.info("create vector engine")
    engine = VectorEngine(df["id"], embeddings)

    # save vector engine
    logger.info("save vector engine")
    engine.save(kwargs["output_filepath"])


@click.command()
@click.argument("sentences_filepath", type=click.Path(exists=True))
@click.argument("embeddings_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
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
    build_vector_db(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
