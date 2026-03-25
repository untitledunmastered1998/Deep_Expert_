#!/bin/bash

output_dir="/data/experiments/OIL/cifar100"
for dkd_coef_weight in 1.0 0.1 0.5 2.0; do
  log_file="${output_dir}/dkd_try_${dkd_coef_weight}.log"
  python main.py \
      --dist_weight 10.0 \
      --temp 4.0 \
      --sigma 0.2 \
      --dkd_coef ${dkd_coef_weight} \
      --exp_name "dkd_try_${dkd_coef_weight}" \
      --run_nums 3 \
      --gpu_id 1 2>&1 | tee "${log_file}"
done