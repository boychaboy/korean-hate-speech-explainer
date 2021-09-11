#!/bin/bash
GPU_ID=$1

python main.py --data_dir ./data/beep/ --bert_model klue/bert-base \
    --max_seq_length 128 --learning_rate 5e-5 --num_train_epochs 5 \
    --early_stop 5 --output_dir runs/beep-klue-bert-base --seed 0 \
    --task_name beep --task_mode hate \
    --do_train --train_batch_size 64 --do_eval --eval_batch_size 256 \
    --gpu_id $GPU_ID
