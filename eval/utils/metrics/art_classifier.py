from transformers import pipeline
from PIL import Image

def init_classifier(device="cuda:0",path="/scratch/nkusumba/t2i_eval"):
    return pipeline('image-classification',model=path,device=device)

def evaluate(classifier, img_paths):
    results = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        results.append(classifier(img,top_k=1000)[:1000])
    return results
