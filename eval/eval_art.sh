#!/usr/bin/env bash
ERASE_ID=42
SD_MODEL_ID="sd_model_v1"
CATEGORY="celeb multiconcept erasure"

export PYTHONPATH=$PYTHONPATH:$(pwd)/data


TARGET_CKPT="erasure_multiconcept_dora_artist2_run2/checkpoint_epoch90/pytorch_lora_weights.safetensors"
NAME="art2_unlearn"
DATA_PATH="data/art2_unlearn.txt"
SAVE_DIR="/scratch/nkusumba/eraseflow_multiconcepts_results/${NAME}/sd_results/gen_${NAME}_${ERASE_ID}_${SD_MODEL_ID}_unlearn_set"
mkdir -p "${SAVE_DIR}"
NSAMPLES=250
# --- run the generation ---
python eval/gen_images_art.py \
  --data "${DATA_PATH}" \
  --num-samples 1 \
  --nsamples "${NSAMPLES}" \
  --erase-id "${ERASE_ID}" \
  --model_type sd \
  --category "${CATEGORY}" \
  --save-dir "${SAVE_DIR}" \
  --target_ckpt "${TARGET_CKPT}" \
  --device cuda:0 \
  --clip_guidance_scale 0

NAME="art2_retain"
DATA_PATH="data/art_retain.txt"
SAVE_DIR="/scratch/nkusumba/eraseflow_multiconcepts_results/${NAME}/sd_results/gen_${NAME}_${ERASE_ID}_${SD_MODEL_ID}_unlearn_set"
mkdir -p "${SAVE_DIR}"
NSAMPLES=25
# --- run the generation ---
python eval/gen_images_art.py \
  --data "${DATA_PATH}" \
  --num-samples 1 \
  --nsamples "${NSAMPLES}" \
  --erase-id "${ERASE_ID}" \
  --model_type sd \
  --category "${CATEGORY}" \
  --save-dir "${SAVE_DIR}" \
  --target_ckpt "${TARGET_CKPT}" \
  --device cuda:0 \
  --clip_guidance_scale 0