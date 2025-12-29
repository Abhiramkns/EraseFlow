import os  
import json
from openai import OpenAI
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from gecko.get_gecko_step1_coverage import get_coverage
from gecko.get_gecko_step2_qa import get_qa
from gecko.get_gecko_step3_vqa import vqa_test



from argparse import ArgumentParser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--file", help="Path to the CSV file containing prompts + other fields", required=True)
    return parser.parse_args()

api_key = os.environ.get("OPENAI_KEY")
client = OpenAI(api_key=api_key)

def load_dataset(data_f):
    # Read CSV file
    dataset = pd.read_csv(data_f)
    # delete index column
    
    if 'Unnamed: 0' in dataset.columns:
        dataset = dataset.drop(columns=['Unnamed: 0'])  
    dataset["nudity_toxicity"] = 0.0
    return dataset

def gecko_test(file_path, save_path):
    dataset = load_dataset(file_path)

    visited_p = set()
    all_questions = []
    for _iter, d in tqdm(dataset.iterrows(), total=len(dataset)):
        prompt = d["prompt"]
        if prompt in visited_p:
            continue
        visited_p.add(prompt)
        types = get_coverage(client, prompt)
        i = 0
        while i < 3:
            try:
                qas = get_qa(client, prompt, types)
                break
            except:
                i += 1
                pass
        if i == 3:
            continue
        all_questions.append({
            "prompt": prompt,
            "questions": []
        })
        for qa in qas:
            question = qa["question"]   
            all_questions[-1]["questions"].append(question)     


    with open(save_path, "w") as f:
        json.dump(all_questions, f)
    
if __name__ == "__main__":
    args    = parse_args()

    name = args.file.split('/')[-1].split('.')[0]
    save_dir = "/".join(args.file.split('/')[:-1])
    save_path = f"{save_dir}/{name}_gecko_qa.json"
    gecko_test(args.file, save_path)