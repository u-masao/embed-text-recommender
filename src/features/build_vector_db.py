import logging

import click
import cloudpickle
import mlflow

from src.models import VectorEngine


def build_vector_db(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load dataset
    with open(kwargs["input_filepath"], "rb") as fo:
        df, embeddings = cloudpickle.load(fo)

    # build vector engine
    engine = VectorEngine(df["id"], embeddings)

    # save vector engine
    engine.save(kwargs["output_filepath"])

    # search similar
    similarities, indices = engine.search(embeddings[0])
    logger.info(f"similarities: {similarities}")
    logger.info(f"indices: {indices}")

    # search similar
    similarities, indices = engine.search(embeddings[:3])
    logger.info(f"similarities: {similarities}")
    logger.info(f"indices: {indices}")

    # lookup ids
    like_ids = [6588884, 6592773]
    like_embeddings = engine.ids_to_embeddings(like_ids)
    logger.info(f"like embeddings: {like_embeddings}")
    logger.info(f"like embeddings shape: {like_embeddings.shape}")


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
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

    # process
    build_vector_db(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
