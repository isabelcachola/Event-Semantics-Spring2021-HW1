'''
Script to upweight training data for unbalanced data
'''
import argparse
import logging
import time
import pandas as pd
from sklearn.utils import resample

def upweight_dataset(infile, outfile, col='LABEL',minority_label=1, majority_label=0):
    df = pd.read_csv(infile)
    n_samples = df['LABEL'].value_counts()[majority_label]
    df_minority = df[df[col]==minority_label]
    df_majoirty = df[df[col]==majority_label]
    df_minority_upsampled = resample(df_minority, replace=True, 
                                    n_samples=n_samples, random_state=1)
    df_upsampled = pd.concat([df_majoirty, df_minority_upsampled])
    print(df_upsampled[col].value_counts())
    df_upsampled.to_csv(outfile, index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('infile')
    parser.add_argument('outfile')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    upweight_dataset(args.infile, args.outfile)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')