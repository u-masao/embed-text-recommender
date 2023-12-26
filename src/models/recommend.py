import logging

import click
import cloudpickle
import mlflow

from src.features.build_features import VectorBuilder


def load_engine(filepath):
    return cloudpickle.load(open(filepath, "rb"))


def recommend(kwargs):
    logger = logging.getLogger(__name__)
    vector_builder = VectorBuilder(kwargs["model_name_or_filepath"])
    engine = load_engine(kwargs["vector_engine_filepath"])

    for sentences in [["飼い犬"]]:
        embeddings = vector_builder.encode(sentences)
        similarities, ids = engine.search(embeddings)

        logger.info(embeddings)
        logger.info(similarities)
        logger.info(ids)


@click.command()
@click.argument("vector_engine_filepath", type=click.Path(exists=True))
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
    recommend(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
