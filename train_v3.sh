export PYTHONPATH=$PYTHONPATH:$(pwd)

export CUDA_VISIBLE_DEVICES=1

# python train.py \
#   --target_prompt "Nudity" \
#   --anchor_prompt "Fully dressed" \
#   --name erasure_nudity \
#   --seed 0 \
#   --use_lora \
#   --lora_rank 4 \
#   --mixed_precision bf16 \
#   --num_epochs 20 \
#   --switch_epoch 20 \
#   --learning_rate 3e-4 \
#   --flow_learning_rate 3e-4 \
#   --guidance_scale 5.0 \
#   --cfg \
#   --eta 1.0 \
#   --beta 2.5 \
#   --save_dir ./checkpoints


python train_multiconcept_v2.py \
  --target_prompt "" \
  --anchor_prompt "" \
  --multiconcept_list data/artist5.json \
  --name erasure_multiconcept_dora_artist5_run1 \
  --seed 0 \
  --use_lora \
  --lora_rank 4 \
  --mixed_precision bf16 \
  --num_epochs 150 \
  --switch_epoch 150 \
  --learning_rate 5e-4 \
  --flow_learning_rate 5e-4 \
  --guidance_scale 5.0 \
  --preserve_weight 0.25 \
  --fnconcept 5 \
  --rnconcept 5 \
  --bz 5 \
  --cfg \
  --eta 1.0 \
  --beta 100 \
  --save_dir /scratch/nkusumba/eraseflow_multiconcept/artist5/checkpoints \
  --use_dora \
  --prior_path data/coco_retain.txt