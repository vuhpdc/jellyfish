#!/bin/bash

: ${root_dir="$HOME/jellyfish/"}
: ${src_path="$root_dir/src/"}
: ${log_path="$root_dir/logs/server/"}
: ${run_mode="RELEASE"}
: ${num_gpus=2}

export PYTHONPATH=${root_dir}

[ -e ${log_path} ] || mkdir -p ${log_path}
ulimit -n 500000

handle_sigint() {
	echo "Signalling SIGINT to child server.."
	kill -INT "$server_pid" 2> /dev/null
}

handle_sigterm() {
	echo "Signalling SIGINT to child server.."
	TERMINATE_FLAG=1
	kill -INT "$server_pid" 2> /dev/null
}

kill_server_child_processes() {
	pkill python3
	for child in $( pgrep -P $1 )
	do
		# First send TERM signal for gracefull exit
		kill -TERM ${child} > /dev/null 2>&1
		sleep 20

		if [[ -d /proc/${child} ]]; then
			echo "Killing child ${child} with KILL"
			kill -9 ${child} > /dev/null 2>&1
		fi
	done

	# Child processes could be defunct now.
	# So kill the parent directly.
	if [[ "" != "$( pgrep -P $1 )" ]];
	then
		echo "Killing server $1 forcefully"
		kill -9 $1 >> /dev/null 2>&1
	fi
}

TERMINATE_FLAG=0
trap handle_sigint SIGINT
trap handle_sigterm SIGTERM

while [ ${TERMINATE_FLAG} -ne 1 ]
do 
	python3 ${src_path}/server/main.py \
	  --weights_dir "${root_dir}/pytorch_yolov4/models/" \
	  --model_config_dir "${root_dir}/pytorch_yolov4/models/cfg" \
	  --n_models 16 \
	  --n_gpus ${num_gpus} \
	  --profiled_dir "${root_dir}/pytorch_yolov4/profiles/" \
	  --run_mode "${run_mode}" \
	  --fps_lcd 5 \
	  --log_path "${log_path}" \
	  --schedule_interval 0.5  \
	  --schedule_min_interval 0.5 \
	  --init_model_number 0 \
	  --active_model_count 5 \
	  > "${log_path}/stdout.log" 2>&1 &

	server_pid=$!
	echo "Server ${server_pid} started!"
	wait "$server_pid"
	echo "Server ${server_pid} finished!"

	## post processing
	# Seperate frame_stats for different client ids
	max_clients=50
	for (( client_id=0; client_id<max_clients; client_id++ ))
	do
		grep "^${client_id}" ${log_path}/frame_path.csv > /dev/null 2>&1
		if [ $? -eq 0 ]; then
		  awk "NR==1 || /^${client_id}/" ${log_path}/frame_path.csv > ${log_path}/frame_stats_${client_id}.csv
	  fi
	done

	# Extra safety for process completion
	kill_server_child_processes ${server_pid}

	# exit
done # while
