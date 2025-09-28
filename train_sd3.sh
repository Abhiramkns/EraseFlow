accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes=1 \
    --main_process_port 29502 \
    trainsd3.py \
    --config eraseflow_config.py:eraseflow_1gpu