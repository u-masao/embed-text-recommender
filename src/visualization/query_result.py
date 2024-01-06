# import streamlit as st
import pandas as pd
import json

import glob
def main():
    runs = []
    for filepath in glob.glob('data/processed/scenario/**/*.json', recursive=True):
        run = json.load(open(filepath,'r'))
        runs.append(run)

    raw_df = pd.DataFrame(runs)
    df = pd.DataFrame()
    df['model_name'] = raw_df['configuration'].map(lambda x: x['embedding_model']['batch_size'])
    df['id'] = pd.DataFrame(raw_df['outputs']).map(lambda x: x['result']['id'])
    df['query'] = pd.DataFrame(raw_df['inputs']).map(lambda x: x['positive_query'])

    print(raw_df.columns)
    print(df)


if __name__ =="__main__":
    main()
