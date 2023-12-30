import logging
from pathlib import Path

import click
import mlflow
import pandas as pd

from src.utils import get_device_info


def build_token_list(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load dataset
    df = pd.read_parquet(kwargs["input_filepath"])

    # output
    Path(kwargs["output_filepath"]).parent.mkdir(exist_ok=True, parents=True)
    df.to_parquet(kwargs["output_filepath"])

    # logging
    log_params = {
        "output.length": len(df),
        "output.columns": df.shape[1],
    }
    mlflow.log_params(log_params)
    logger.info(log_params)
    logger.info(f"output dataframe: \n{df}")
    logger.info(f"output columns: \n{df.columns}")


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):
    """
    Word2Vec の学習用にトークン一覧を作成します。
    """
    # init logging
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiment("features")
    mlflow.start_run(run_name="mlflow_run_name")

    # log cli options
    logger.info(f"args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})
    mlflow.log_params(get_device_info())

    # process
    build_token_list(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
