# Evaluation Scripts README

## Block-wise CSD Similarity Evaluation

Evaluates generated images using CSD-CLIP by computing block-wise cosine
similarity against reference embeddings.

------------------------------------------------------------------------

### CSD Checkpoint

To run the below script, you need to provide a CSD model checkpoint.  
You can download it from [here](https://drive.google.com/file/d/1FX0xs8p-C7Ob-h5Y4cUhTeOepHzXv_46/view).

### Usage

``` bash
python metrics/art_csd_eval.py --dir outputs --ckpt_path csd_ckpt.pt --refer_path ref_embeds.pt
```

Outputs JSON score per model folder.

## Grouped CSD Similarity

Supports multiple generated images per reference index and keeps the
best similarity per group.

### Usage

``` bash
python metrics/art_csd_eval_for_udatk.py --dir outputs --ckpt_path csd_ckpt.pt --refer_path ref_embeds.pt
```

------------------------------------------------------------------------
## Nudity Rate Evaluation (NudeNet)

Detects NSFW content and computes nudity statistics.

### Usage (root mode)

``` bash
python nudity_eval.py --mode root --root outputs/model --threshold 0.6
```

------------------------------------------------------------------------

## GECKO VQA Concept Probing

Runs visual QA with GPT-4o to measure concept leakage.

### Usage

``` bash
python gecko_vqa.py --file dataset.csv --questions_file questions.json
```

------------------------------------------------------------------------
