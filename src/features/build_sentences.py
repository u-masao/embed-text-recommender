import logging

import click
import mlflow
import pandas as pd


def build_sentences(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load dataset
    df = pd.read_parquet(kwargs["input_filepath"])

    # limit sentence size
    if kwargs["limit_sentence_size"] > 0:
        df = df.head(kwargs["limit_sentence_size"])

    # make features
    df["sentence"] = df["title"] + "\n" + df["content"]

    # output dataframe
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
@click.option("--limit_sentence_size", type=int, default=0)
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

    # process
    build_sentences(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
