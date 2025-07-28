export PYTHONPATH=$PYTHONPATH:$(pwd)

export CUDA_VISIBLE_DEVICES=1

# python train.py \
#   --target_prompt "Van Gogh" \
#   --anchor_prompt "Artist" \
#   --name erasure_vangogh \
#   --seed 1 \
#   --use_lora \
#   --lora_rank 4 \
#   --mixed_precision bf16 \
#   --num_epochs 20 \
#   --switch_epoch 0 \
#   --learning_rate 5e-4 \
#   --flow_learning_rate 5e-4 \
#   --guidance_scale 5.0 \
#   --cfg \
#   --eta 1.0 \
#   --beta -5.0 \
#   --save_dir /scratch/nkusumba/eraseflow_abalation/beta/-5/vangogh_checkpoints

# python train.py \
#   --target_prompt "Van Gogh" \
#   --anchor_prompt "Artist" \
#   --name erasure_vangogh \
#   --seed 1 \
#   --use_lora \
#   --lora_rank 4 \
#   --mixed_precision bf16 \
#   --num_epochs 20 \
#   --switch_epoch 0 \
#   --learning_rate 5e-4 \
#   --flow_learning_rate 5e-4 \
#   --guidance_scale 5.0 \
#   --cfg \
#   --eta 1.0 \
#   --beta -2.5 \
#   --save_dir /scratch/nkusumba/eraseflow_abalation/beta/-2.5/vangogh_checkpoints

# python train.py \
#   --target_prompt "Van Gogh" \
#   --anchor_prompt "Artist" \
#   --name erasure_vangogh \
#   --seed 1 \
#   --use_lora \
#   --lora_rank 4 \
#   --mixed_precision bf16 \
#   --num_epochs 20 \
#   --switch_epoch 0 \
#   --learning_rate 5e-4 \
#   --flow_learning_rate 5e-4 \
#   --guidance_scale 5.0 \
#   --cfg \
#   --eta 1.0 \
#   --beta 0 \
#   --save_dir /scratch/nkusumba/eraseflow_abalation/beta/0/vangogh_checkpoints

# python train.py \
#   --target_prompt "Van Gogh" \
#   --anchor_prompt "Artist" \
#   --name erasure_vangogh \
#   --seed 1 \
#   --use_lora \
#   --lora_rank 4 \
#   --mixed_precision bf16 \
#   --num_epochs 20 \
#   --switch_epoch 0 \
#   --learning_rate 5e-4 \
#   --flow_learning_rate 5e-4 \
#   --guidance_scale 5.0 \
#   --cfg \
#   --eta 1.0 \
#   --beta 5.0 \
#   --save_dir /scratch/nkusumba/eraseflow_abalation/beta/5.0/vangogh_checkpoints


# ###########################################################################################


# python train.py \
#   --target_prompt "Caravaggio" \
#   --anchor_prompt "Artist" \
#   --name erasure_vangogh \
#   --seed 1 \
#   --use_lora \
#   --lora_rank 4 \
#   --mixed_precision bf16 \
#   --num_epochs 20 \
#   --switch_epoch 0 \
#   --learning_rate 5e-4 \
#   --flow_learning_rate 5e-4 \
#   --guidance_scale 5.0 \
#   --cfg \
#   --eta 1.0 \
#   --beta -5.0 \
#   --save_dir /scratch/nkusumba/eraseflow_abalation/beta/-5/vangogh_checkpoints

# python train.py \
#   --target_prompt "Caravaggio" \
#   --anchor_prompt "Artist" \
#   --name erasure_caravaggio \
#   --seed 1 \
#   --use_lora \
#   --lora_rank 4 \
#   --mixed_precision bf16 \
#   --num_epochs 20 \
#   --switch_epoch 0 \
#   --learning_rate 5e-4 \
#   --flow_learning_rate 5e-4 \
#   --guidance_scale 5.0 \
#   --cfg \
#   --eta 1.0 \
#   --beta -2.5 \
#   --save_dir /scratch/nkusumba/eraseflow_abalation/beta/-2.5/caravaggio_checkpoints

# python train.py \
#   --target_prompt "Caravaggio" \
#   --anchor_prompt "Artist" \
#   --name erasure_caravaggio \
#   --seed 1 \
#   --use_lora \
#   --lora_rank 4 \
#   --mixed_precision bf16 \
#   --num_epochs 20 \
#   --switch_epoch 0 \
#   --learning_rate 5e-4 \
#   --flow_learning_rate 5e-4 \
#   --guidance_scale 5.0 \
#   --cfg \
#   --eta 1.0 \
#   --beta 0 \
#   --save_dir /scratch/nkusumba/eraseflow_abalation/beta/0/caravaggio_checkpoints

# python train.py \
#   --target_prompt "Caravaggio" \
#   --anchor_prompt "Artist" \
#   --name erasure_caravaggio \
#   --seed 1 \
#   --use_lora \
#   --lora_rank 4 \
#   --mixed_precision bf16 \
#   --num_epochs 20 \
#   --switch_epoch 0 \
#   --learning_rate 5e-4 \
#   --flow_learning_rate 5e-4 \
#   --guidance_scale 5.0 \
#   --cfg \
#   --eta 1.0 \
#   --beta 5.0 \
#   --save_dir /scratch/nkusumba/eraseflow_abalation/beta/5.0/caravaggio_checkpoints

for INDEX in 0 5.0 10.0 15.0 20.0 25.0 50.0 100.0
do 
  python train.py \
  --target_prompt "Caravaggio" \
  --anchor_prompt "Artist" \
  --name erasure_caravaggio \
  --seed 1 \
  --use_lora \
  --lora_rank 4 \
  --mixed_precision bf16 \
  --num_epochs 20 \
  --switch_epoch 0 \
  --learning_rate 5e-4 \
  --flow_learning_rate 5e-4 \
  --guidance_scale 5.0 \
  --cfg \
  --eta 1.0 \
  --beta $INDEX \
  --save_dir /scratch/nkusumba/eraseflow_abalation/beta/$INDEX/caravaggio_checkpoints
done

for INDEX in 0 5.0 10.0 15.0 20.0 25.0 50.0 100.0
do 
  python train.py \
  --target_prompt "Van Gogh" \
  --anchor_prompt "Artist" \
  --name erasure_vangogh \
  --seed 1 \
  --use_lora \
  --lora_rank 4 \
  --mixed_precision bf16 \
  --num_epochs 20 \
  --switch_epoch 0 \
  --learning_rate 5e-4 \
  --flow_learning_rate 5e-4 \
  --guidance_scale 5.0 \
  --cfg \
  --eta 1.0 \
  --beta $INDEX \
  --save_dir /scratch/nkusumba/eraseflow_abalation/beta/$INDEX/vangogh_checkpoints
done