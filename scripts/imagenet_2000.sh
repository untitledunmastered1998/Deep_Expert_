#!/bin/bash

output_dir="/data/experiments/OIL/imagenet/buffer2000"
log_file="${output_dir}/weight_1e1_sdl_1e3_runs3_buffer_2000.log"
python main.py \
      --dataset imagenet_1k \
      --n_tasks 10 \
      --buffer_size 2000 \
      --compensate "off" \
      --dist_weight 0.0 \
      --sdl_weight 0.001 \
      --temp 4.0 \
      --sigma 0.2 \
      --exp_name "weight_1e1_sdl_1e3_runs3_buffer_2000" \
      --run_nums 3 \
      --gpu_id 1 2>&1 | tee "${log_file}"
