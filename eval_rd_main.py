import pandas as pd
import argparse
import os
import sys
import csv
from ast import literal_eval
from sentence_transformers import SentenceTransformer,util
import numpy as np
import ast
import torch
from pathlib import Path
import re

# Evaluation metric
def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])
    
    
def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)
    
def acc(ground_truth, prediction):

    accu_1 = 0.
    accu_10 = 0.
    accu_100 = 0. 
    length = len(ground_truth)
    pred_rank = []   
    
   
    for i in range(length):
        try:
            pred_rank.append(prediction[i].index(ground_truth[i]))
        except ValueError:             
            pred_rank.append(1000)
        
        if ground_truth[i] in prediction[i][:100]:
            accu_100 += 1
            if ground_truth[i] in prediction[i][:10]:
                accu_10 += 1
                if ground_truth[i] == prediction[i][0]:
                    accu_1 += 1

    accuracy_1 = (accu_1 / length) * 100
    accuracy_10 = (accu_10 / length) * 100
    accuracy_100= (accu_100 / length) * 100
    
    median_rank = np.median(pred_rank)
    std_rank = np.sqrt(np.var(pred_rank))

    return accuracy_1, accuracy_10, accuracy_100, median_rank, std_rank

if __name__ == '__main__':
  # Create the parser
  parser = argparse.ArgumentParser()

  parser.add_argument('-d','--dataset',help='The dataset',required=True)
  parser.add_argument('-t', '--dataset_type', help='The dataset type (3dex or hill)', default="hill")
  parser.add_argument('-r', '--rd_method', help='The reverse dictionary method', default="gear")
    
  args = parser.parse_args()
  
  # Read the dataset
  dataset = pd.read_csv(args.dataset, engine='python', na_values = [''], keep_default_na=False)
  
  if (args.dataset_type == "3dex"):
  
      if (args.rd_method == "llm"):
          dataset['HITS'] = dataset['HITS'].apply(lambda x: [int(i) for i in ast.literal_eval(x)])
          hits_list = dataset['HITS'].tolist()
  
      if (args.rd_method == "gear"): 
          dataset['EMBEDDING_HITS'] = dataset['EMBEDDING_HITS'].apply(lambda x: [int(i) for i in ast.literal_eval(x)])
          hits_list = dataset['EMBEDDING_HITS'].tolist()
          
      mrr = mean_reciprocal_rank(hits_list)
      k_values = [1, 3, 5]
      precision = {k: np.mean([precision_at_k(hits, k) for hits in hits_list]) for k in k_values}
      print(f"mrr = {mrr}, precision = {precision}")
    
  else:
      if (args.rd_method == "llm"):
          dataset['PREDICTED_TERMS'] = dataset['PREDICTED_TERMS'].apply(ast.literal_eval)
          accuracy_1, accuracy_10, accuracy_100, median_rank, std_rank =  acc(dataset['word'], dataset['PREDICTED_TERMS'])
  
      if (args.rd_method == "gear"): 
          dataset['PREDICTED_TERMS_BY_EMBEDDINGS'] = dataset['PREDICTED_TERMS_BY_EMBEDDINGS'].apply(ast.literal_eval)
          accuracy_1, accuracy_10, accuracy_100, median_rank, std_rank =  acc(dataset['word'], dataset['PREDICTED_TERMS_BY_EMBEDDINGS'])
      
      print(f"acc@1/10/100 = {accuracy_1}/{accuracy_10}/{accuracy_100}, median_rank = {median_rank}, std_rank = {std_rank}")    
      
