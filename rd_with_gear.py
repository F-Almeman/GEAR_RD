import pandas as pd
import openai
import json
import argparse
import csv
from pathlib import Path
import sys
import ast
import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from InstructorEmbedding import INSTRUCTOR
import torch


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--dataset_split', help='The dataset', required=True)
    parser.add_argument('-t', '--dataset_type', help='The dataset type (3dex or hill)', default="hill")
    parser.add_argument('-m', '--model', help='Model type', default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('-k', '--number_terms', help='Number of nearest terms', type=int, default=100)
    parser.add_argument('-o', '--output_path', help='Path to the output file', required=True)
   
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Read the dataset
    df = pd.read_csv(args.dataset_split, engine='python', na_values=[''], keep_default_na=False)
    df['PREDICTED_TERMS'] = df['PREDICTED_TERMS'].apply(ast.literal_eval)
    
    # Convert word column to a list of unique words
    if (args.dataset_type == "3dex"):
        df['TERMS'] = df['TERMS'].apply(ast.literal_eval)
        unique_terms = set(term for sublist in df['TERMS'] for term in sublist)
        terms_list = list(unique_terms)
    
    else:
        terms = df['word'].unique()
        terms_list = terms.tolist() 
     
    # Get words embeddings   
    model = SentenceTransformer(args.model)
    terms_embeddings = model.encode(terms_list, convert_to_tensor=True)

    hits_column = []
    pred_terms =[]

    for idx in range(len(df)):

        # Get embeddings of predicted terms
        pred_terms_embed = model.encode(df.PREDICTED_TERMS.iloc[idx], convert_to_tensor=True)
        
        # Compute the average embedding  
        avg_embed = torch.mean(pred_terms_embed, dim=0)
        avg_embed = avg_embed.to(device)

        # Compute cosine similarities between the average embedding and all terms embeddings
        sim = util.cos_sim(avg_embed, terms_embeddings)

        top_indices = torch.topk(sim, args.number_terms).indices.squeeze(0).tolist()
        
        top_terms = [terms_list[i] for i in top_indices]
        
        pred_terms.append(top_terms)
     
        if args.dataset_type == "3dex":
            hits = [1 if t in df.TERMS.iloc[idx] else 0 for t in top_terms]
        else:
            hits = [1 if t == df.word.iloc[idx] else 0 for t in top_terms]
                
        hits_column.append(hits)
            
    df["PREDICTED_TERMS_BY_EMBEDDINGS"] =  pred_terms
    df["EMBEDDING_HITS"] =  hits_column
    output_file = os.path.join(args.output_path, f"{Path(args.dataset_split).stem}_top_{args.number_terms}_nearest_neighbors_sbert.csv")
    df.to_csv(output_file, index=False)

