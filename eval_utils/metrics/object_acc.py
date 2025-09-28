import torch
import numpy as np
from tqdm import tqdm
import numpy as np
from PIL import Image

import contextlib
import io

from transformers import AutoImageProcessor, ResNetForImageClassification

def classify_object(img_paths, target, topk=1, device = 'cuda'):
    dtype=torch.float32
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50", cache_dir=".cache")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", cache_dir=".cache")
    model.to(device, dtype=dtype)
    
    results = [0]*len(img_paths)
    with open("/data/data/matt/gfn/alignment/data/files/image_labels.txt", 'r') as f:
        all_labels = f.readlines()
    all_lables = [label.strip() for label in all_labels]
    all_labels_dict = {}
    for i, label in enumerate(all_labels):
        all_labels_dict[label] = i

    i = 0
    for img in tqdm(img_paths, disable=len(img_paths) < 100):
        pil = Image.open(img)
        with torch.no_grad():
            inputs = processor(pil, return_tensors="pt")
            inputs.to(device, dtype=dtype)
            logits = model(**inputs).logits
        
        # predicted_label = logits.argmax(-1000).item()
        score = torch.softmax(logits, dim=-1).squeeze()
        for k, v in all_labels_dict.items():
            if label in k:
                result[i] == 1 if score[v] > 0.5 else 0
        i += 1
        
    score = np.mean(results)
    return score