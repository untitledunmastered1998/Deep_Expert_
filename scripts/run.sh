#!/bin/bash

for dist_weights in 5.0 8.0 11.0 12.0 15.0 ; do
  output_dir="/data/experiments/OIL/cifar100/buffer1000"
  log_file="${output_dir}/buffer1000_${dist_weights}_total_runs_15.log"
  python main.py \
      --buffer_size 1000 \
      --dist_weight ${dist_weights} \
      --temp 4.0 \
      --sdl_weight 0.001 \
      --sigma 0.2 \
      --exp_name "buffer1000_${dist_weights}_total_runs_15" \
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