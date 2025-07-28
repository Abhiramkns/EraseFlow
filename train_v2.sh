export PYTHONPATH=$PYTHONPATH:$(pwd)

export CUDA_VISIBLE_DEVICES=0

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
  --multiconcept_list data/celeb1_adamdriver.json \
  --name erasure_multiconcept_adamdriver_celeb1_run6 \
  --model_id /scratch/nkusumba/eraseflow_multiconcept/celeb1/checkpoints/erasure_multiconcept_celeb1_run5/celeb1_model \
  --seed 0 \
  --use_lora \
  --lora_rank 4 \
  --mixed_precision bf16 \
  --num_epochs 1000 \
  --switch_epoch 1000 \
  --learning_rate 3e-4 \
  --flow_learning_rate 3e-4 \
  --guidance_scale 5.0 \
  --preserve_weight 0.25 \
  --fnconcept 5 \
  --rnconcept 5 \
  --bz 5 \
  --cfg \
  --eta 1.0 \
  --beta 100 \
  --save_dir /scratch/nkusumba/eraseflow_multiconcept/celeb1/checkpoints 