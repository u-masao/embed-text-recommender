# Embed Text Recommender

## 目的

このソフトウェアはテキストに対する検索システムです。
検索キーワードだけでなく好みの文書 ID を与えることで、好みに近い文書を検索することが可能です。
単独の検索システムというよりも、推薦モデルに近い使い方を想定しています。

複数の埋め込みモデルの動作を比較することで、求めるタスクに対して最適なモデルすることが目的です。


## 動作概要

以下のプロセス構造で実行します。

- make repro で dvc パイプラインを実行

  - dvc でパイプラインを順番に実行

    - データ取得

    - Word2Vec 辞書の作成、モデルの学習

    - SentenceTransformer モデルのダウンロード

    - テキスト文書の埋め込み作成

    - 埋め込み Database 作成

    - 検索サンプル実行

- make ui で Gradio Web UI を実行

  - Gradio で UI を実行
  
    - 設定をロード

    - モデルとテキストの埋め込みをロード

    - 検索キーワード等の入力画面を表示

    - 検索キーワード等の埋め込みを計算

    - 検索キーワード等の埋め込みに距離が近いテキスト文書を検索

    - 距離が近いテキスト文書を表示する



## 動作環境


以下の環境で動作確認しています。

- OS: Ubuntu 22.04 LTS

- Python: Version 3.10


## インストール方法


以下のコマンドを実行します。


- ワーキングディレクトリの作成と移動

```
$ mkdir -p ~/work
$ cd ~/work
```

- リポジトリ取得

```
$ git clone https://github.com/u-masao/embed-text-recommender.git
$ cd embed-text-recommender
```

- [poetry のインストール](https://python-poetry.org/docs/)（お好みの方法でインストールしてください）

```
$ python3 -m pip install --user pipx
$ pipx ensurepath
$ source ~/.bashrc
$ pipx install poetry
```

- 仮想環境を作成

```
$ poetry install
```

## 実行方法と利用方法

- 埋め込みベクトルデータベースの作成

```
$ make repro
```

- UI バックエンドの実行

```
$ make ui
```

- UI の利用

  - ブラウザで以下の URL を開く

    http://localhost:7860/




## 実行結果のサンプル

（作成予定）

## 設定変更


- 処理パイプラインの設定

  - params.yaml を修正

- UI の設定

  - ui.yaml を修正
