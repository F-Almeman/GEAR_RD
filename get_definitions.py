import pandas as pd
import argparse
import os
import sys
import csv
import numpy as np
import random
from ast import literal_eval

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-d','--dataset_file',help='The main dataset (term, definition, example, source)',required=True)
  parser.add_argument('-o','--output_path',help='Path to the output file',required=True)

  args = parser.parse_args()
        
  # Read the dataset file
  dataset = pd.read_csv(args.dataset_file, engine='python', na_values = [''], keep_default_na=False)

  # Convert the string representations of lists back to actual lists
  dataset['DATASET_NAME'] = dataset['DATASET_NAME'].apply(eval)

  # Create a dataframe of (definition, [list of terms], [list of examples] [list of sources])
  new_dataset = {}

  for index, row in dataset.iterrows():
    definition = row['DEFINITION']
    word = row['TERM']
    if pd.isna(row['EXAMPLE']):
      examples = []
    else:
      examples = row['EXAMPLE']
    source_list = row['DATASET_NAME']

    # Check if the definition is already in the new_dataset dictionary
    if definition in new_dataset:
      new_dataset[definition]['TERMS_LIST'].append(word)
      new_dataset[definition]['EXAMPLES'].append(examples)
      new_dataset[definition]['SOURCES'].append(source_list)
    else:
      new_dataset[definition] = {'TERMS_LIST': [word], 'EXAMPLES': [examples], 'SOURCES': [source_list] }

  # Create a list of tuples from the new_data dictionary
  new_data_list = [(key, value['TERMS_LIST'], value['EXAMPLES'], value['SOURCES']) for key, value in new_dataset.items()]

  # Create a new dataframe from the list of tuples
  new_df = pd.DataFrame(new_data_list, columns=['DEFINITION', 'TERMS', 'EXAMPLES', 'SOURCES'])

  new_df.to_csv(os.path.join(args.output_path, "definitions_dataset.csv"), index = False, header=True)
