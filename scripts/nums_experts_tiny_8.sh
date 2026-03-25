for bf in 1000 2000 4000 ; do
  output_dir="/data/experiments/OIL/tiny/100tasks/ablation/nums_experts"
  log_file="${output_dir}/reverse_6experts_${bf}.log"
  python main.py \
      --nums_expert 6 \
      --dataset tiny_imagenet \
      --n_tasks 100 \
      --buffer_size ${bf} \
      --dist_weight 20.0 \
      --temp 4.0 \
      --buffer_batch_size 32 \
      --sdl_weight 0.001 \
      --sigma 0.2 \
      --exp_name "reverse_6experts_${bf}" \
      --run_nums 1 \
      --compensate "on" \
      --gpu_id 1 2>&1 | tee "${log_file}"
done