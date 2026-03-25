for bf in 10000 4000 2000 ; do
  output_dir="/data/experiments/OIL/tiny/100tasks/ablation"
  log_file="${output_dir}/wo_kd_total_runs_3_${bf}.log"
  python main.py \
      --nums_expert 1 \
      --dataset tiny_imagenet \
      --n_tasks 100 \
      --buffer_size ${bf} \
      --dist_weight 10.0 \
      --temp 4.0 \
      --sdl_weight 0.001 \
      --sigma 0.2 \
      --exp_name "wo_kd_total_runs_3_${bf}" \
      --run_nums 1 \
      --compensate "off" \
      --gpu_id 1 2>&1 | tee "${log_file}"
done