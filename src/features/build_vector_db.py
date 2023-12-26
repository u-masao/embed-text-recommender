import logging

import click
import cloudpickle
import faiss
import mlflow
import numpy as np


class VectorEngine:
    def __init__(self, ids, embeddings):
        self.ids = ids
        self.embeddings = embeddings
        self.dimension = embeddings.shape[1]

        # normalize
        l2norms = np.linalg.norm(embeddings, axis=1, ord=2).flatten()
        self.normalized_embeddings = embeddings / l2norms

        # init faiss
        dimension = embeddings.size[1]
        index = faiss.IndexFlatIP(dimension)
        index.add_with_ids(self.normailzed_embeddings, ids)

    def search(self, query_embedding, top_n: int = 5):
        similarities, indices = self.index.search(
            np.array(query_embedding), top_n
        )

        return similarities, indices


def build_vector_db(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load dataset
    with open(kwargs["input_filepath"], "rb") as fo:
        df, embeddings = cloudpickle.load(fo)
    logger.info(df)
    logger.info(embeddings)

    # build vector engine
    engine = VectorEngine(df["id"], embeddings)

    # search similar
    similarities, indices = engine.search(embeddings[0])
    print(similarities)
    print(indices)


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
    build_vector_db(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
