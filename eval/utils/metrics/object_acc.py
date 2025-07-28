import torch
import numpy as np
from tqdm import tqdm
import numpy as np

import contextlib
import io

def classify_object(img_paths, target, device = 'cuda'):
    resnet50 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

    resnet50.eval().to(device)
    results = [0]*len(img_paths)
    i = 0
    for img in tqdm(img_paths, disable=len(img_paths) < 100):
        batch = torch.cat([utils.prepare_input_from_uri(img)]).to(device)
        with torch.no_grad():
            output = torch.nn.functional.softmax(resnet50(batch), dim=1)
        
        with contextlib.redirect_stdout(io.StringIO()):
            result = utils.pick_n_best(predictions=output, n=1000)[0]
        for j in range(len(result)):
            if target in result[j][0]:
                s = result[j][-1].strip('%')
                number_float = float(s)
                number_float /= 100
                results[i] = 1 if number_float > 0.5 else 0
                break
            else:
                results[i] = 0
        i += 1
        
    score = np.mean(results)
    return score