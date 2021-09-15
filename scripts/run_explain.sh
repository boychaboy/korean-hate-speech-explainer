#!/bin/bash
GPU_ID=$1
CUDA_LAUNCH_BLOCKING=1 python main.py --data_dir ./data/beep/ --bert_model klue/bert-base \
    --max_seq_length 128 --output_dir runs/beep-klue-bert-base \
    --lm_dir runs/lm_beep --task_name beep --task_mode hate --eval_batch_size 1 \
    --hiex --hiex_add_itself --output_filename hiex.pkl \
    --nb_range 3 --sample_n 3 --algo soc --do_eval --explain \
    --gpu_id $GPU_ID --fp16
