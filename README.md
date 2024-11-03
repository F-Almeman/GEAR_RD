# GEAR_RD

This repository is created for the work **GEAR: A Simple \textsc{Generate, Embed, Average and Rank} Approach for Unsupervised Reverse Dictionary** that is submitted to COLING 2025.


# 1. Datasts 

## 1.1 3D-EX

From 3D-EX dataset ([Almeman et al., 2023](https://aclanthology.org/2023.ranlp-1.8/)), a unified resource containing several dictionaries in the format of (Term, Definition, Example, Source), we retrieved all the definitions along with their corresponding terms, examples and sources, creating <definition, [list_of_terms_defined_by_that_definition], [list_of_examples],  [list_of_sources]> tuples. [dataset.csv](https://drive.google.com/uc?export=download&id=1TdVx9Pk3SQ16vWkr8WBi6SLpKMV9tIm6)

```
python3 get_definitions.py -d 3dex.csv -o datasets
```
-d: input file (dataset) <br/>
-o: output folder 

## Splitting 
Two splits are created: Random_Split (train, validation, and test) and Source_Split. Source_Split splits the dataset based on the specified source ('MultiRD', 'CODWOE', 'Webster\'s Unabridged', 'Urban', 'Wikipedia', 'WordNet', 'Wiktionary', 'Hei++', 'CHA', 'Sci-definition')

```
python3 split_dataset.py -d datasets/dataset.csv -o datasets -s "WordNet"
```
-d: input file  <br/>
-o: output folder <br/>
-s: split type (default = "random" )

## 1.2 Hill dataset
[Hill test sets](https://drive.google.com/file/d/1ihfElRULa6bg_jpwzeHSJEC2KQc_w25p/view) (seen, unseen, description) from [Hill et al., (2016)](https://arxiv.org/pdf/1504.00548).


# 2. Experiments

## 2.1 LLM-based approach 

This script is to generate terms by a LLM to be used for the reverse dictionary task. Given a datset and its desciption and some examples, k terms will be generated based on the specified prompt. 

```
python3 rd_with_llm.py -s "datasets/WordNet_test.csv" -t "3dex" -m "gpt-4o-mini" -k 5 -o datasets  -d "WordNet" -p 2
```
-s: dataset split file <br/>
-t: dataset type which are 3dex or hill (default="hill") <br/>
-m: llm model (default="gpt-4o-mini") <br/>
-k: number of generated terms (default=5) <br/>
-o: output folder <br/>
-d: dataset name (default="Hill seen") <br/>
-p: prompt type (1: base prompt 1, 2: base prompt 2, 3: reasoning prompt) 

## 2.2 GEAR (LLM-based + Embeddings) 

This script is to generate terms by a LLM to be used for the reverse dictionary task. Given a datset and its desciption and some examples, k terms will be generated based on the specified prompt. 

```
python3 rd_with_gear.py -s "datasets/WordNet_test.csv" -t "3dex" -m "gpt-4o-mini" -k 5 -o datasets  -d "WordNet" -p 2
```
-s: dataset input file (the output file from LLM-based approach) <br/>
-m: embeddings model (default='sentence-transformers/all-MiniLM-L6-v2') <br/>
-o: output folder <br/>


# 3. Evaluation


