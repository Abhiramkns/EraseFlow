# Evaluation Scripts README

# 1. Block-wise CSD Similarity Evaluation

This script evaluates generated images using **CSD-CLIP** by computing
block-wise cosine similarity against a set of reference embeddings.

### CSD Checkpoint

To run the below scripts, you need to provide a CSD model checkpoint.  
You can download it from [here](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view).

### Usage

``` bash
python metrics/art_csd_eval.py   --dir outputs   --ckpt_path csd_ckpt.pt   --refer_path ref_embeds.pt
```

------------------------------------------------------------------------

# 2. Reference Embeddings Format & Generation

Both CSD similarity evaluation scripts rely on a tensor of precomputed
**reference embeddings** representing semantic targets.

Instead of using arbitrary images, we recommend generating reference
embeddings from a **base (unaltered) diffusion model** using the same
evaluation prompts applied during testing.

This ensures similarity comparisons are aligned with the generative
distribution and avoids dataset bias.

------------------------------------------------------------------------

## Recommended Workflow

1.  Use the evaluation prompt set\
2.  Generate one reference image per prompt using the base model\
3.  Encode those images using CSD-CLIP\
4.  Normalize and save embeddings

------------------------------------------------------------------------

## Expected Tensor Shape

    [N, D]

Where:

-   **N** = number of evaluation prompts\
-   **D** = CSD-CLIP embedding dimension

The number of reference embeddings must exactly match:

-   Script 1 → number of generated images per model folder\
-   Script 2 → number of indexed concept groups

------------------------------------------------------------------------

## Example: Generating Reference Embeddings

``` python
import torch
import torch.nn.functional as F
from CSD.model import CSD_CLIP
from CSD.loss_utils import transforms_branch0
from PIL import Image

model = CSD_CLIP("vit_large", content_proj_head="default").cuda().eval()

imgs = [transforms_branch0(Image.open(p).convert("RGB")) for p in ref_image_paths]
imgs = torch.stack(imgs).cuda()

with torch.no_grad():
    feats = model(imgs)[0]
    feats = F.normalize(feats, dim=-1)

torch.save(feats.float(), "ref_embeds.pt")
```

Important:

-   Use the same CSD checkpoint for encoding and evaluation\
-   Apply identical preprocessing (`transforms_branch0`)\
-   Maintain consistent prompt ordering\
-   Always normalize embeddings before saving

------------------------------------------------------------------------

# 3. Grouped CSD Similarity

This variant supports multiple generated images per reference index and
retains the **best similarity score per group** before averaging.

### Usage

``` bash
python metrics/art_csd_eval_for_udatk.py   --dir outputs   --ckpt_path csd_ckpt.pt   --refer_path ref_embeds.pt
```

------------------------------------------------------------------------

# 4. Nudity Rate Evaluation (NudeNet)

Computes nudity statistics using NudeNet predictions.

### Usage (Root Mode)

``` bash
python metrics/compute_nudity_rate.py   --mode root   --root outputs/model   --threshold 0.6
```

------------------------------------------------------------------------

# 5. GECKO VQA Concept Probing

Runs Yes/No visual question answering using GPT-4o to measure concept
presence or leakage.


### Usage

``` bash
python metrics/vqa.py   --file dataset.csv   --questions_file questions.json
```

Ensure your API key is set:

``` bash
export OPENAI_KEY=your_api_key
```
