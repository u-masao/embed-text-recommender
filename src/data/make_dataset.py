import logging

import click
import mlflow
import pandas as pd
from datasets import load_dataset


def make_dataset(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load dataset
    dataset = load_dataset(
        kwargs["dataset_name_or_path"],
        train_ratio=0.8,
        validation_ratio=0.1,
        seed=kwargs["random_state"],
        shuffle=True,
    )

    # to pandas dataframe
    results = []
    for name, data in dataset.items():
        df = pd.DataFrame(data).assign(name=name)
        results.append(df)
        logger.info(f"dataset: {name}: {data}")
    result_df = pd.concat(results)

    # output
    result_df.to_parquet(kwargs["output_filepath"])
    log_params = {
        "output.length": len(result_df),
        "output.columns": result_df.shape[1],
    }
    mlflow.log_params(log_params)
    logger.info(log_params)


@click.command()
@click.argument("dataset_name_or_path", type=click.Path())
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--random_state", type=int, default=1234)
def main(**kwargs):
    # init logging
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiment("make_dataset")
    mlflow.start_run(run_name="mlflow_run_name")

    # log cli options
    logger.info(f"args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})

    # process
    make_dataset(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
