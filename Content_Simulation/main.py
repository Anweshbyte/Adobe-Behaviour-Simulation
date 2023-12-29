## Importing packages
import argparse
import pandas as pd
from DataCleaning import master_clean
from master import master_try
from utils import get_prompt
import os
from captioning import get_captions

### Loading the dataset
def main(dataset_path):
    # dataset_path = "content_simulation_train.xlsx"
    df = pd.read_excel(dataset_path)
    df = master_clean(df)

    df['caption'] = get_captions(df)
    df['prompt'] = df.apply(get_prompt,axis = 1)

    df = master_try(df)

    results = df['id']
    results['prediction'] = df['content']
    try:
        os.mkdir("outputs")
    except FileExistsError:
        # Handle the case where the directory already exists
        pass
    results.to_excel("outputs/results.xlsx",index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset and make predictions.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset in Excel format.")
    args = parser.parse_args()

    main(args.dataset_path)