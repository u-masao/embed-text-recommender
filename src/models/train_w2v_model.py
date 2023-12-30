import logging
from pathlib import Path

import click
import cloudpickle
import mlflow
from gensim.models import word2vec

from src.utils import get_device_info


def train_w2v_model(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load dataset
    word_list = cloudpickle.load(open(kwargs["input_filepath"], "rb"))

    # learn
    model = word2vec.Word2Vec(
        word_list, size=100, min_count=5, window=5, iter=100
    )

    # output
    Path(kwargs["output_w2v_filepath"]).parent.mkdir(
        exist_ok=True, parents=True
    )
    model.save(kwargs["output_w2v_filepath"])
    Path(kwargs["output_kv_filepath"]).parent.mkdir(
        exist_ok=True, parents=True
    )
    model.wv.save(kwargs["output_kv_filepath"])

    # logging
    log_params = {}
    mlflow.log_params(log_params)
    logger.info(log_params)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_w2v_filepath", type=click.Path())
@click.argument("output_kv_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):
    """
    Word2Vec のモデルを作成します
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
    train_w2v_model(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
