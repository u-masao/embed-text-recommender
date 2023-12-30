import logging
from pathlib import Path

import click
import cloudpickle
import mlflow
import pandas as pd
from janome.tokenizer import Tokenizer
from tqdm import tqdm

from src.utils import get_device_info


def build_token_list(kwargs):
    # init logging
    logger = logging.getLogger(__name__)

    # load dataset
    df = pd.read_parquet(kwargs["input_filepath"])
    df["sentence"] = df["title"] + "\n" + df["content"]

    # limit sentence size
    if kwargs["limit_sentence_size"] > 0:
        df = df.head(kwargs["limit_sentence_size"])

    # concat sentence
    full_text = "\n".join(df["sentence"].values)

    # init tokenizer
    tokenizer = Tokenizer()

    def extract_words(text):
        tokens = tokenizer.tokenize(text)
        words = []
        for token in tokens:
            if token.part_of_speech.split(",")[0] in ["名詞", "動詞"]:
                words.append(token.base_form)

        return words

    sentences = full_text.split("。")
    word_list = [extract_words(sentence) for sentence in tqdm(sentences)]

    # output
    Path(kwargs["output_filepath"]).parent.mkdir(exist_ok=True, parents=True)
    with open(kwargs["output_filepath"], "wb") as fo:
        cloudpickle.dump(word_list, fo)

    # logging
    logger.info(f"word_list head 5: {word_list[:5]}")
    log_params = {
        "output.length": len(word_list),
    }
    mlflow.log_params(log_params)
    logger.info(log_params)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.option("--limit_sentence_size", type=int, default=0)
@click.option("--mlflow_run_name", type=str, default="develop")
def main(**kwargs):
    """
    Word2Vec の学習用にトークン一覧を作成します。
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
    build_token_list(kwargs)

    # cleanup
    mlflow.end_run()
    logger.info("complete process")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
