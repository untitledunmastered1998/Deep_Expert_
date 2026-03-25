#!/bin/bash

output_dir="/data/experiments/OIL/tiny"
for buffer_nums in 2000 4000 10000; do
  log_file="${output_dir}/same_dims_all_pred_dist_weight_10_runs15_${buffer_nums}.log"
  python main.py \
      --dataset tiny_imagenet \
      --buffer_size ${buffer_nums} \
      --dist_weight 0.1 \
      --temp 4.0 \
      --sigma 0.2 \
      --exp_name "same_dims_all_pred_dist_weight_10_runs15_${buffer_nums}" \
      --run_nums 15 \
      --gpu_id 0 2>&1 | tee "${log_file}"
done

#log_file="${output_dir}/same_dims_all_pred_dist_weight_10_replace_old.log"
#python main.py \
#    --dist_weight 10.0 \
#    --temp 4.0 \
#    --sigma 0.2 \
#    --exp_name "same_dims_all_pred_dist_weight_10_replace_old" \
#    --run_nums 3 \
#    --gpu_id 0 2>&1 | tee "${log_file}"