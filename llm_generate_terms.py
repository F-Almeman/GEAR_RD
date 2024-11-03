import pandas as pd
import openai
import json
import argparse
import csv
from pathlib import Path
import sys
import ast
import os
import time


def generate_terms(definition, llm_model, k, dictionary, description, examples):
    try:
        if args.prompt == 1:
            response = openai.ChatCompletion.create(
                model=llm_model,   
                messages=[
                    {
                    "role": "system",
                    "content": "You are a dictionary writing assistant. You are knowledgeable about word senses, word meanings, and can provide accurate definitions."
                    },
                    {
                    "role": "user",
                    "content": f"""Given the definition {definition}, generate a list of {k} terms defined by that definition assuming they are in {dictionary} dictionary.  
                    Only give me a list back, do not generate any other text.

                    {description}

                    The returned list should follow the following conditions:
                    - Terms should be ordered or ranked so the first term is the most related to the definition.
                    - In a JSON object of the form {{ "terms": ["term_1", "term_2", ... ] }}.
                    - All terms should be in lowercase.

                    Example:

                    INPUT:
                    "A piece of furniture for sitting."

                    OUTPUT:
                    {{ "terms": ["chair", "stool", "bench", "sofa", "couch"] }}
                    """
                    }
                  ]
            )
            
        if args.prompt == 2:
            response = openai.ChatCompletion.create(
                model=llm_model,   
                messages=[
                    {
                    "role": "system",
                    "content": "You are a dictionary writing assistant. You are knowledgeable about word senses, word meanings, and can provide accurate definitions."
                    },
                    {
                    "role": "user",
                    "content": f"""Given the definition '{definition}', generate a list of {k} terms defined by that definition assuming they are in {dictionary} dictionary.  
                    Only give me a list back, do not generate any other text.

                    {description}

                    These are some examples of definitions and terms in this dictionary: {examples}

                    The returned list should follow the following conditions:
                    - Terms should be ordered or ranked so the first term is the most related to the definition.
                    - In a JSON object of the form {{ "terms": ["term_1", "term_2", ... ] }}.
                    - All terms should be in lowercase.

                    Example:

                    INPUT:
                    "A piece of furniture for sitting."

                    OUTPUT:
                    {{ "terms": ["chair", "stool", "bench", "sofa", "couch"] }}
                     """
                    }
                  ]
            )
        
        # Parse the result from the response
        result_json = json.loads(response['choices'][0]['message']['content'].strip())
        terms = result_json.get('terms', [])
        if len(terms) < k:
            terms += [""] * (k - len(terms))
        result_json['terms'] = terms
        return result_json
    
    except openai.error.RateLimitError as e:
        time.sleep(9)
        return generate_terms(definition, llm_model, k, dictionary, description)  # Retry the request
    
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing response: {e}")
        print(f"Definition: {definition}")
        return None

def generate_terms_examples(definition, llm_model, k, dictionary, description, examples):
    try:
        response = openai.ChatCompletion.create(
            model=llm_model,   
            messages=[
            {
             "role": "system",
             "content": "You are a dictionary writing assistant. You are knowledgeable about word senses, word meanings, and can provide accurate definitions."
            },
            {
            "role": "user",
            "content": f"""Given the definition '{definition}', generate a list of {k} terms defined by that definition assuming they are in the {dictionary} dictionary.
            Only give me a list back, do not generate any other text.
            
            {description}
                        
            These are some examples of definitions and terms in this dictionary: {examples}

            For each term, provide an example usage in a sentence that matches the style and scope of {dictionary}.
                        
            The returned list should follow these conditions:
            - Terms should be ranked, with the first term being the most related to the definition.
            - All terms and examples should be in lowercase.
            - Return the terms and examples in a JSON object of the form:
                {{
                "terms": [
                       {{ "term": "term_1", "example": "example_1" }},
                       {{ "term": "term_2", "example": "example_2" }},
                         ...
                         ]
                }}
                
            Example:

            INPUT:
            "A piece of furniture for sitting."

            OUTPUT:
            {{
              "terms": [
                {{
                  "term": "chair",
                  "example": "he sat on the chair and opened his book."
                }},
                {{
                  "term": "stool",
                  "example": "she perched on the stool at the bar."
                }},
                {{
                  "term": "bench",
                  "example": "they rested on the bench after their walk."
                }},
                {{
                  "term": "sofa",
                  "example": "the family gathered on the sofa to watch TV."
                }},
                {{
                  "term": "couch",
                  "example": "he stretched out on the couch to take a nap."
                }}
              ]
             }}
             """
            }
          ]
        )

        # Parse the result from the response
        result_json = json.loads(response['choices'][0]['message']['content'].strip())
        terms = result_json.get('terms', [])

        # Ensure k terms and k examples are generated
        if len(terms) != k:
            if len(terms) > k:
                # Truncate to k terms
                terms = terms[:k]
                
            elif len(terms) < k:
                # Fill with empty placeholders if fewer than k terms
                missing_count = k - len(terms)
                empty_terms = [{"term": "", "example": ""} for _ in range(missing_count)]
                terms += empty_terms

        return {"terms": terms}

    except openai.error.RateLimitError as e:
        time.sleep(9)
        return generate_terms_examples(definition, llm_model, k, dictionary, description, examples)  # Retry the request
    
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing response: {e}")
        print(f"Definition: {definition}")
        return None 

                        
def save_progress(df, output_path):
    """Save the current progress to a CSV file."""
    df.to_csv(output_path, index=False, header=True)
    print(f"Progress saved to {output_path}")


def load_checkpoint(temp_output_path):
    """Load checkpoint if it exists."""
    if os.path.exists(temp_output_path):
        print(f"Checkpoint found. Resuming from {temp_output_path}")
        df_checkpoint = pd.read_csv(temp_output_path)
        return df_checkpoint
    else:
        return None

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--dataset_split', help='The dataset', required=True)
    parser.add_argument('-t', '--dataset_type', help='The dataset type (3dex or hill)', default="hill")
    parser.add_argument('-m', '--model', help='The llm model', default="gpt-4o-mini")
    parser.add_argument('-k', '--number_terms', help='Number of generated terms', type=int, default=5)
    parser.add_argument('-o', '--output_path', help='Path to the output file', required=True)
    parser.add_argument('-d', '--dictionary', help='dictionary name', default="random")
    parser.add_argument('-p', '--prompt', help='prompt number', type=int, default=1)
    parser.add_argument('--save_interval', help='Save progress after this many rows', type=int, default=10)

    args = parser.parse_args()


    # Read the dataset
    df = pd.read_csv(args.dataset_split, engine='python', na_values=[''], keep_default_na=False)
    
    if (args.dataset_type == "3dex"):
        df['TERMS'] = df['TERMS'].apply(ast.literal_eval)
    
    #####################################################################################
    # To save and load checkpoints. in case of errors 
    # Initialize columns if not present
    if 'HITS' not in df.columns:
        df['HITS'] = pd.Series(dtype=object)
    if 'PREDICTED_TERMS' not in df.columns:
        df['PREDICTED_TERMS'] = pd.Series(dtype=object)
    if args.prompt == 3:    
        if 'PREDICTED_EXAMPLES' not in df.columns:
            df['PREDICTED_EXAMPLES'] = pd.Series(dtype=object)
    
    # Output file path for saving progress
    temp_output_path = os.path.join(args.output_path, f"{args.dataset_type}_{Path(args.dataset_split).stem}_{args.model}_prompt_{args.prompt}_checkpoint.csv")

    # Load checkpoint if exists
    df_checkpoint = load_checkpoint(temp_output_path)
    if df_checkpoint is not None:
        # Merge the checkpoint data with the original data (to account for unprocessed rows)
        df.update(df_checkpoint)
        start_idx = len(df_checkpoint)  # Start from the next row after the last processed
    else:
        start_idx = 0  # Start from the first row if no checkpoint exists

    hits_column = df['HITS'].tolist() if 'HITS' in df.columns else []
    pred_terms_column = df['PREDICTED_TERMS'].tolist() if 'PREDICTED_TERMS' in df.columns else []
    if args.prompt == 3:
        pred_ex_column = df['PREDICTED_EXAMPLES'].tolist() if 'PREDICTED_EXAMPLES' in df.columns else []
    #####################################################################################
    
    '''
    openai.api_key = <YOUR KEY>
    

    dictionary_description = <DICTIONARY DESCRIPTION>
    
    dictionary_examples = <DICTIONARY EXAMPLES>
    '''
    
    openai.api_key = "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"
    

    dictionary_description = """WordNet is an electronic lexical database for English that organises words in groups of synonyms called synsets. 
    synset is described by its definition, surface forms (lemmas), examples of usage (where available), and the relations between synsets.
     WN’s primary use in NLP is as a sense inventory."""
     
    dictionary_examples = """EXAMPLE 1: (Definition: 'broad in the beam', terms: ['beamy']]),
    EXAMPLE 2: (Definition: 'providing sophisticated amusement by virtue of having artificially (and vulgarly) mannered or banal or sentimental qualities', terms:['camp', 'campy'])"""
    

    for idx in range(start_idx, len(df)):
    
        if args.dataset_type == "3dex":
            definition = df.DEFINITION.iloc[idx].strip()
            gold_terms = df.TERMS.iloc[idx]
        
        else:
            definition = df.definitions.iloc[idx].strip()
            gold_term = df.word.iloc[idx]
        
        if args.prompt == 3:
            result = generate_terms_examples(definition, args.model, args.number_terms, args.dictionary, dictionary_description, dictionary_examples)
            
            # Extract terms and examples from the result
            if result is not None:
                pred_terms = [item['term'] for item in result['terms']]
                pred_examples = [item['example'] for item in result['terms']]
            else:
                pred_terms = ["", "", "", "", ""]
                pred_examples = ["", "", "", "", ""]
                
            
            # Save examples
            df.at[idx, "PREDICTED_EXAMPLES"] = pred_examples
            
           
        else:
            result = generate_terms(definition, args.model, args.number_terms, args.dictionary, dictionary_description, dictionary_examples)
            pred_terms = result.get('terms', []) if result is not None else ["", "", "", "", ""]

        # Create the hits list
        if args.dataset_type == "3dex":
            hits = [1 if t in gold_terms else 0 for t in pred_terms]
        else:
            hits = [1 if t == gold_term else 0 for t in pred_terms]
        

        # Save terms and hits
        df.at[idx, "HITS"] = hits  
        df.at[idx, "PREDICTED_TERMS"] = pred_terms      
    
        # Save progress every N rows (set by args.save_interval)
        if (idx + 1) % args.save_interval == 0:
            save_progress(df.iloc[:idx + 1], temp_output_path)

    # Save final output
    final_output_path = os.path.join(args.output_path, f"{args.dataset_type}_{Path(args.dataset_split).stem}_{args.model}_prompt_{args.prompt}_llm_rd_dataset.csv")
    save_progress(df, final_output_path)
    print(f"Final output saved to {final_output_path}")

