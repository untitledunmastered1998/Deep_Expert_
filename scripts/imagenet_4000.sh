#!/bin/bash

output_dir="/data/experiments/OIL/imagenet/buffer4000"
log_file="${output_dir}/weight_5e1_sdl_1e3_runs3_buffer_4000.log"
python main.py \
      --dataset imagenet_1k \
      --n_tasks 10 \
      --buffer_size 4000 \
      --compensate "off" \
      --dist_weight 0.0 \
      --sdl_weight 0.001 \
      --temp 4.0 \
      --sigma 0.2 \
      --exp_name "weight_5e1_sdl_1e3_runs3_buffer_4000" \
      --run_nums 3 \
      --gpu_id 0 2>&1 | tee "${log_file}"
