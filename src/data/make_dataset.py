import logging

import click
import mlflow
from datasets import load_dataset


def make_dataset(kwargs):
    dataset = load_dataset(
        kwargs["dataset_name_or_path"],
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_state=kwargs["random_state"],
        suffle=True,
    )

    print(dataset)


@click.command()
@click.argument("dataset_name_or_path", type=click.Path())
@click.argument("output_filepath", type=click.Path())
@click.option("--mlflow_run_name", type=str, default="develop")
@click.option("--random_state", type=int, default=1234)
def main(**kwargs):
    # init logging
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiments("make_dataset")
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
