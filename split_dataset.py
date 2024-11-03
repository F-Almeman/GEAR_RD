import pandas as pd
import argparse
import os
import sys
import csv
import numpy as np
import random
import ast
from ast import literal_eval

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-d','--dataset_file',help='The dataset file',required=True)
  parser.add_argument('-o','--output_path',help='Path to output files',required=True)
  parser.add_argument('-s','--split_type',help='Split type (random or source split)',default="random")

  args = parser.parse_args()
        
  # Read the dataset file
  dataset = pd.read_csv(args.dataset_file, na_values = [''], keep_default_na=False)
  dataset['TERMS'] = dataset['TERMS'].apply(ast.literal_eval)
  dataset['EXAMPLES'] = dataset['EXAMPLES'].apply(ast.literal_eval)
  dataset['SOURCES'] = dataset['SOURCES'].apply(ast.literal_eval)
  
  # Source Split
  if args.split_type != "random":
    dataset = dataset[dataset['SOURCES'].apply(lambda x: any(args.split_type in sublist_element for sublist in x for sublist_element in sublist))]
    
    # Update 'TERMS' and 'EXAMPLES' 'SOURCES' columns based on the filter
    dataset['TERMS'] = dataset.apply(lambda row: [terms_list for terms_list, sources_list in zip(row['TERMS'], row['SOURCES']) if any(args.split_type in sublist for sublist in sources_list)], axis=1)
    dataset['EXAMPLES'] = dataset.apply(lambda row: [examples_list for examples_list, sources_list in zip(row['EXAMPLES'], row['SOURCES']) if any(args.split_type in sublist for sublist in sources_list)], axis=1)
    dataset['SOURCES'] = dataset['SOURCES'].apply(lambda x: [sources_list for sources_list in x if any(args.split_type in sublist for sublist in sources_list)])

  train, valid, test = \
              np.split(dataset.sample(frac=1, random_state=42),
                       [int(.6*len(dataset)), int(.8*len(dataset))])

  train.to_csv(os.path.join(args.output_path, f"{args.split_type}_train.csv"), header = True, index = False)
  valid.to_csv(os.path.join(args.output_path, f"{args.split_type}_valid.csv"), header = True, index = False)
  test.to_csv(os.path.join(args.output_path, f"{args.split_type}_test.csv"), header = True, index = False)
