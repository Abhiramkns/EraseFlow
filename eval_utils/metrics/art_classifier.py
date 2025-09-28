from transformers import pipeline
from PIL import Image

def init_classifier(device="cuda:7",path="/data/data/matt/gfn/alignment/style_classifier/results/checkpoint-2800"):
    return pipeline('image-classification',model=path,device=device)

def evaluate(classifier, img_paths):
    results = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        results.append(classifier(img,top_k=1000)[:1000])
    return results