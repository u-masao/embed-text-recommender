import logging

import click
import mlflow
import yaml

from src.models import QueryHandler
from src.utils import get_device_info


def query_simulator(kwargs):
    # init logger
    logger = logging.getLogger(__name__)

    # load scenario
    scenarios = yaml.safe_load(
        open(kwargs["scenario_filepath"], "r", encoding="utf-8")
    )["scenarios"]
    logger.info(f"{scenarios=}")

    # init QueryHandler
    handler = QueryHandler(kwargs["scenario_filepath"])
    handler.config['log_dir'] = kwargs['log_output_dir']

    for scenario in scenarios:
        print(scenario)
        print(
            handler.search(
                scenario.get("positive_query", ""),
                scenario.get("positive_query_blend_ratio", 1.0),
                scenario.get("negative_query", ""),
                scenario.get("negative_query_blend_ratio", 1.0),
                scenario.get("like_ids", ""),
                scenario.get("like_blend_ratio", 1.0),
                scenario.get("dislike_ids", ""),
                scenario.get("dislike_blend_ratio", 1.0),
                scenario.get("top_n", 20),
            )
        )


@click.command()
@click.argument("search_engine_filepath", type=click.Path(exists=True))
@click.argument("log_output_dir", type=click.Path())
@click.option(
    "--embedding_model_string",
    type=str,
    default="SentenceTransformer/oshizo/sbert-jsnli-luke-japanese-base-lite",
)
@click.option("--chunk_method", type=str, default="chunk_split")
@click.option(
    "--scenario_filepath",
    type=click.Path(exists=True),
    default="scenario.yaml",
)
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):
    """
    クエリ文字列に類似するテキストを計算する。

    Parameters
    ------
    kwargs: Dict[str, any]
        CLI オプション
        - embedding_model_string
            埋め込み作成モデルの指定(Storategy / モデル名)
        - mlflow_run_name
            MLflow の run_name
    """
    # init logging
    logger = logging.getLogger(__name__)
    logger.info("start process")
    mlflow.set_experiment("simulation")
    mlflow.start_run(run_name="mlflow_run_name")

    # log cli options
    logger.info(f"args: {kwargs}")
    mlflow.log_params({f"args.{k}": v for k, v in kwargs.items()})
    mlflow.log_params(get_device_info())

    # process
    query_simulator(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
