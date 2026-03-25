#!/bin/bash

output_dir="/data/experiments/OIL/tiny/100tasks"
for buffer_nums in 2000 4000 10000; do
  log_file="${output_dir}/same_dims_all_pred_dist_weight_0.1_runs15_${buffer_nums}.log"
  python main.py \
      --dataset tiny_imagenet \
      --n_tasks 100 \
      --buffer_size ${buffer_nums} \
      --dist_weight 0.1 \
      --temp 4.0 \
      --sigma 0.2 \
      --exp_name "same_dims_all_pred_dist_weight_0.1_runs15_${buffer_nums}" \
      --run_nums 15 \
      --gpu_id 1 2>&1 | tee "${log_file}"
done
