#!/bin/bash

: ${root_dir="$HOME/jellyfish/"}
: ${src_path="$root_dir/src/"}
: ${log_path="$root_dir/logs/simulations/comparision"}

export PYTHONPATH=${root_dir}
[ -e ${log_path} ] || mkdir -p ${log_path}

algo_comparision="milp_vs_sa"
GPUS="2 4"
CLIENTS_FACTOR="4 6 8"

for num_gpus in ${GPUS}
do
	for i in ${CLIENTS_FACTOR}
	do
		num_clients=$(( ${num_gpus} * ${i} ))
		echo "Experiment: ${algo_comparision}_${num_gpus}_${num_clients}"

		python3 ${src_path}/simulation/main_compare.py \
		  	--num_clients ${num_clients} \
		  	--num_gpus ${num_gpus} \
		  	--num_models 16 \
		  	--profiled_dir "${root_dir}/pytorch_yolov4/profiles/" \
		  	--fps_lcd 5 \
		  	--max_batch_size 8 \
        	--use_profiled_values \
		  	> "${log_path}/${algo_comparision}_${num_gpus}_${num_clients}.log" 2>&1
                  
        mv "${num_gpus}_${num_clients}.npy" "${log_path}/${algo_comparision}_${num_gpus}_${num_clients}.npy"
	done
done

