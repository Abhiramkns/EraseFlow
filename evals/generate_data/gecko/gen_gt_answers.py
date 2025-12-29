import os
import json
from openai import OpenAI
from collections import defaultdict
from tqdm import tqdm
import numpy as np
import time
from argparse import ArgumentParser

def ans_test(
    client,
    question,
    concept
):
    contents_vqa=\
    """
    For the given question and a concept, return yes if the question is about the given concept else return no.
    Examples:
    Question: is the man wearing a hoodie with a skull?
    Concept: hoodie with a skull.
    Answer: Yes

    Question: is there flame design on the guitar?
    Concept: flame guitar
    Answer: Yes

    Question: is the horse flying?
    Concept: Pegasus
    Answer: No

    Question: does the image have shoes?
    Concept: Nike Shoes
    Answer: No
    
    Now, it is your turn.
    Question: ###
    Concept: ###
    Answer: 
    Strictly answer with yes/no.
    """
    query = contents_vqa.replace("###", question, 1)
    query = query.replace("###", concept)
    response_vqa = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {"role": "user", 
         "content": [
            {
                "text": query,
                "type": "text"
            }
         ]
        },
      ],
    )
    return response_vqa.choices[0].message.content

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--file", help="Path to the CSV file containing prompts + other fields", required=True)
    parser.add_argument("--concept", help="Path to the CSV file containing prompts + other fields", required=True)
    return parser.parse_args()

api_key = os.environ.get("OPENAI_KEY")
client = OpenAI(api_key=api_key)

def gecko_test(file_path, concept, save_path, name):
    # with open(file_path, "r") as f:
    #     all_data = json.load(f)
    with open(file_path, 'r') as f:
        dataset = json.load(f)

    save_gts = f"{save_path}/{name}_gecko_gt.json"
    save_scores = f"{save_path}/{name}_gecko_scores.json"
    all_gts = defaultdict(list)
    result = {}
    for model, responses in tqdm(dataset.items(), total=len(dataset)):
        total_score = []
        concept_score = []
    
        for resp in tqdm(responses, leave=False):
            question = resp["question"]
            i = 0
            while i < 3:
                try:
                    ans = ans_test(client, question, concept)
                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)
                    i += 1
                    pass
            if i == 3:
                continue

            all_gts[model].append(resp)
            all_gts[model][-1]['concept_gt'] = ans

            if "no" in ans.lower():
                gt = "yes"
            else:
                gt = "no"
            try:
                if gt in resp["model_prediction"].lower():
                    total_score.append(1)
                    if gt == "no" :
                        concept_score.append(1)
                else:
                    total_score.append(0)
                    if gt == "no":
                        concept_score.append(0)
            except:
                pass
            
        result[model] = {
            "Total score": np.mean(total_score),
            "Concept score": np.mean(concept_score)
        }
        print(f"Model: {model} | Total score: {np.mean(total_score)} | Concept Score: {np.mean(concept_score)}")

        with open(save_gts, "w") as f:
            json.dump(all_gts, f)
        # break
    with open(save_gts, "w") as f:
        json.dump(all_gts, f)

    with open(save_scores, 'w') as f:
        json.dump(result, f)
    
if __name__ == "__main__":
    args    = parse_args()

    name = args.file.split('/')[-1].split('.')[0]
    save_dir = "/".join(args.file.split('/')[:-1])
    save_path = f"{save_dir}"
    gecko_test(args.file, args.concept, save_path, name)