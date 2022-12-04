#!/bin/bash

: ${root_dir="$HOME/jellyfish/"}
: ${src_path="${root_dir}/src/"}
: ${profile_path="${root_dir}/pytorch_yolov4/profiles/latency/"}
: ${log_path="${root_dir}/logs/profiler/"}
: ${run_mode="RELEASE"}
: ${num_gpus=2}
: ${num_models=16}

FRAME_SIZES=(128 160 192 224 256 288 320 352 384 416 448 480 512 544 576 608)
export PYTHONPATH=${root_dir}

# clean logs repo
rm -r ${log_path}/*
[ -e ${log_path} ] || mkdir -p ${log_path}
[ -e ${profile_path} ] || mkdir -p ${profile_path}

ulimit -n 500000

for (( model_number=0; model_number < num_models; model_number++ ))
do
	echo "Profiling model number ${model_number}"

	python3 ${src_path}/server/profiler/main.py \
	  --weights_dir "${root_dir}/pytorch_yolov4/models/" \
	  --model_config_dir "${root_dir}/pytorch_yolov4/models/cfg" \
	  --n_models ${num_models} \
	  --n_gpus ${num_gpus} \
	  --run_mode "${run_mode}" \
	  --fps_lcd 5 \
	  --log_path "${log_path}" \
	  --init_model_number ${model_number} \
	  --max_batch_size 12 \
	  --dataset_dir "${root_dir}/datasets/coco/val2017/images"\
	  --gt_annotations_path "${root_dir}/datasets/coco/val2017/annotations/instances_val2017.json" \
	  --total_profile_iter 2000 \
	  --profile_dir "${log_path}" \
	  > ${log_path}/stdout_${model_number}.log 2>&1

	# Merge and save gpu profiles
	frame_size=${FRAME_SIZES[$model_number]}
	cp "${log_path}/profiles_gpu_0/profile_latency_${frame_size}.txt" "${profile_path}/profile_latency_${frame_size}.txt"
	for (( gpu_number=1; gpu_number < num_gpus; gpu_number++ ))
	do
		sed '1'd "${log_path}/profiles_gpu_${gpu_number}/profile_latency_${frame_size}.txt" >> "${profile_path}/profile_latency_${frame_size}.txt"
	done
done
