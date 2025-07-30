huggingface-cli login --token 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

python dist_run.py \
    --file "coco10k.csv" \
    --concept "clip_score" \
    --gpus 0,1,2,3,4,5,6,7 \
    --save_dir './clip_images' \
    --ckpts_path all_ckpts/all_ablation_ckpts.json

python cal_clip_score.py \
    --input_path './clip_images' \

python cal_fid.py \
    --input_path './clip_images/clip_score' \
    --real_folder './coco10k_images' \