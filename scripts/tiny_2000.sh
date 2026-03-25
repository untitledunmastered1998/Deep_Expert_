#!/bin/bash

output_dir="/data/experiments/OIL/tiny/100tasks"
log_file="${output_dir}/weight_1e1_sdl_1e3_runs15_buffer_2000.log"
python main.py \
      --dataset tiny_imagenet \
      --n_tasks 100 \
      --buffer_size 2000 \
      --dist_weight 0.1 \
      --sdl_weight 0.001 \
      --temp 4.0 \
      --sigma 0.2 \
      --exp_name "weight_1e1_sdl_1e3_runs15_buffer_2000" \
      --run_nums 15 \
      --gpu_id 1 2>&1 | tee "${log_file}"
