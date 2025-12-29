import os  
import json
import base64
import numpy as np
from openai import OpenAI
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--file", help="Path to the CSV file containing prompts + img_path", required=True)
    parser.add_argument("--questions_file", help="Path to the json file containing questions for VQA test", required=True)
    return parser.parse_args()

api_key = os.environ.get("OPENAI_KEY")
client = OpenAI(api_key=api_key)

def encode_image(img_path):
    with open(img_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_image

def vqa_test(
    client,
    img_path,
    question
):
    encoded_image = encode_image(img_path)
    contents_vqa=\
    """
    Question: ###
    Strictly answer with yes/no. Answer yes only if you are 100\% sure.
    """
    query = contents_vqa.replace("###", question)
    response_vqa = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "user", 
         "content": [
            {
                "text": query,
                "type": "text"
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
            }
         ]
        },
      ],
    )
    return response_vqa.choices[0].message.content

def gecko_vqa_test(file_path, questions_file, save_path):
    dataset = pd.read_csv(file_path)
    with open(questions_file, "r") as f:
        questions = json.load(f)
    qas = []
    score = []
    concept_score = []
    non_concept_score = []
    for idx, d in tqdm(dataset.iterrows(), total=len(dataset)):
        img_path = d["img_path"]
        prompt = d["prompt"]
        for ele in questions[idx]:
            gt = ele["ground_truth"]
            answer = vqa_test(client, img_path, ele['question'])
            if gt.lower() in answer.lower():
                score.append(1)
                if gt == "No.":
                    concept_score.append(0)
                else:
                    non_concept_score.append(1)
            else:
                score.append(0)
                if gt == "No.":
                    concept_score.append(1)
                else:
                    non_concept_score.append(0)
            qas.append({
                "prompt": prompt,
                "question": ele['question'],
                "img_path": img_path,
                "answer": answer
            })

    score = np.mean(score)
    results = {
        "total_score": np.mean(score),
        "concept_score": np.mean(concept_score),
        "non_concept_score": np.mean(non_concept_score)
    }
    with open(os.path.join(save_path, 'gecko_score.json'), 'w') as f:
        json.dump(results, f)

    with open(os.path.join(save_path, "gecko_qa.json"), "w") as f:
        json.dump(qas, f)
    
if __name__ == "__main__":
    args    = parse_args()

    save_dir = "/".join(args.file.split('/')[:-1])
    gecko_vqa_test(args.file, args.questions_file, save_dir)