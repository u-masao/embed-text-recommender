import logging

import click
import cloudpickle
import mlflow
import pandas as pd
from sentence_transformers import SentenceTransformer


class VectorBuilder:
    def __init__(self, model_name_or_filepath):
        self.model_name_or_filepath = model_name_or_filepath
        self.model = SentenceTransformer(model_name_or_filepath)

    def encode(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings


def embedding(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load dataset
    df = pd.read_parquet(kwargs["input_filepath"])

    # make features
    df["sentence"] = df["title"] + "\n" + df["content"]

    # embedding
    builder = VectorBuilder(kwargs["model_name_or_filepath"])
    embeddings = builder.encode(df["sentence"])

    # output
    with open(kwargs["output_filepath"], "wb") as fo:
        cloudpickle.dump([df, embeddings], fo)

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
@click.option(
    "--model_name_or_filepath",
    type=str,
    default="oshizo/sbert-jsnli-luke-japanese-base-lite",
)
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

    # process
    embedding(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
