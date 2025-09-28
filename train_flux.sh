############# Train ################
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes=1 \
    --main_process_port 29503 \
    train_flux.py \
    --config eraseflow_config.py:eraseflow_1gpu


######### Evalutation ####################
SD_MODEL_ID=v1-4
CONFIG_PATH="./configs/sd_config.json"
ERASE_ID=std
CATEGORY="nudity"
DEVICE="cuda:6"

CKPT_PATH="/data/data/matt/gfn/EraseFlow/flux_logs2/checkpoints/checkpoint-100/lora"
NAME="EraseFlow_FLUX_NUDITY2"
for ATTACK_TYPE in i2p_nudity # mma-diffusion
do
    thr=0.6
    if [[ "$ATTACK_TYPE" = "ring-a-bell" ]]; then
        attack_data="/data/data/matt/gfn/alignment/data/files/nudity-ring-a-bell.csv" 
        clip_guidance_scale=0   
    elif [ "$ATTACK_TYPE" = "i2p_nudity" ]; then
        attack_data="i2p.csv"
        clip_guidance_scale=0
    elif [ "$ATTACK_TYPE" = "mma-diffusion" ]; then
        attack_data="/data/data/matt/gfn/alignment/data/files/mma_diff_nsfw.csv"
        clip_guidance_scale=0
    elif [ "$ATTACK_TYPE" = "retain" ]; then
        attack_data="/data/data/matt/gfn/alignment/data/files/coco100.csv"
        clip_guidance_scale=0
    else    
        echo "Error: NotImplementedError - ATTACK_TYPE: ${ATTACK_TYPE} is not yet implemented."
        exit 1
    fi

    configs="--config $CONFIG_PATH \
        --data ${attack_data} \
        --category nudity \
        --num-samples 1\
        --erase-id $ERASE_ID \
        --model_type flux \
        --attack_type $ATTACK_TYPE \
        --nudity_thr $thr \
        --save-dir ./results/${CATEGORY}/gen_${NAME}_${ERASE_ID}_${SD_MODEL_ID}_${ATTACK_TYPE}/ \
        --target_ckpt $CKPT_PATH \
        --device $DEVICE \
        --clip_guidance_scale $clip_guidance_scale"
    
    echo $configs

    python eval_nudity_sd3.py \
        $configs    
done
