#!/bin/bash
GPU_ID=$1
python main.py --data_dir ./data/beep/ --bert_model klue/bert-base \
    --max_seq_length 128 --output_dir runs/buyu_${buyu} \
    --lm_dir runs/lm_beep --task_name buyu --eval_batch_size 1 \
    --hiex --hiex_add_itself --output_filename hiex.pkl \
    --nb_range 5 --sample_n 3 --algo soc --do_eval --test --explain \
    --gpu_id $GPU_ID --only_positive --buyu ${buyu}
