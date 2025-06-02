export PYTHONPATH=$PYTHONPATH:$(pwd)

export CUDA_VISIBLE_DEVICES=0

python train.py \
  --target_prompt "Nudity" \
  --anchor_prompt "Fully dressed" \
  --name erasure_nudity \
  --seed 0 \
  --use_lora \
  --lora_rank 4 \
  --mixed_precision bf16 \
  --num_epochs 20 \
  --switch_epoch 20 \
  --learning_rate 3e-4 \
  --flow_learning_rate 3e-4 \
  --guidance_scale 5.0 \
  --cfg \
  --eta 1.0 \
  --beta 2.5 \
  --save_dir ./checkpoints
