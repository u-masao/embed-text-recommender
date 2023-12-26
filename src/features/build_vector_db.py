import logging

import click
import cloudpickle
import faiss
import mlflow
import numpy as np


class VectorEngine:
    def __init__(self, ids, embeddings):
        """
        VectorEngine を初期化する。
        ID一覧と埋め込みを利用して内部のDBに登録する。

        Parameters
        ------
        ids: List[int]
            ID
        embeddings: numpy.ndarray
            埋め込み表現

        Returns
        ------
        None
        """
        self.ids = ids
        self.embeddings = embeddings
        self.dimension = embeddings.shape[1]

        # normalize
        l2norms = np.linalg.norm(embeddings, axis=1, ord=2)
        self.normalized_embeddings = (embeddings.T / l2norms).T

        # init faiss
        base_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap(base_index)

        # add vectors to faiss
        self.index.add_with_ids(self.normalized_embeddings, ids)

    def search(self, query_embeddings, top_n: int = 5):
        query_embeddings = query_embeddings.reshape(-1, self.dimension)
        l2norms = np.linalg.norm(query_embeddings, axis=1, ord=2)
        normalized_query_embeddings = (query_embeddings.T / l2norms).T
        similarities, indices = self.index.search(
            normalized_query_embeddings, top_n
        )
        return similarities, indices

    def _find_ids_indices(self, ids):
        results = []
        for id in ids:
            if id in self.ids:
                results.append(self.ids.index(id))
        return results

    def ids_to_embeddings(self, ids: List[int], normalized: bool = False):
        if normalized:
            return self.normalized_embeddings[self._find_ids_indices(ids)]
        else:
            return self.embeddings[self._find_ids_indices(ids)]


def build_vector_db(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load dataset
    with open(kwargs["input_filepath"], "rb") as fo:
        df, embeddings = cloudpickle.load(fo)

    # build vector engine
    engine = VectorEngine(df["id"], embeddings)
    cloudpickle.dump(engine, open(kwargs["output_filepath"], "bw"))

    # search similar
    similarities, indices = engine.search(embeddings[0])
    logger.info(f"similarities: {similarities}")
    logger.info(f"indices: {indices}")

    # search similar
    similarities, indices = engine.search(embeddings[:3])
    logger.info(f"similarities: {similarities}")
    logger.info(f"indices: {indices}")


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
