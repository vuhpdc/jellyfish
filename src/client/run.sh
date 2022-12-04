#!/bin/bash

if [ $# -ne 5 ]; then
  echo "Usage: ./run.sh <server_host_ip> <my_id> <slo> <fps> <video_name>"
  echo "Example: run.sh 192.168.1.100 host_0 100 15 dds/trafficcam_1.mp4"
  exit
fi

my_id=$2
slo=$3
fps=$4
video_name=$5

: ${root_dir="$HOME/jellyfish/"}
: ${src_path="$root_dir/src/"}
: ${video_file="${root_dir}/datasets/${video_name}"}
: ${image_file="${root_dir}/datasets/coco/val2017/image_list_10.txt"}
: ${model_server_host="$1"}
: ${stats_fname="frame_stats.csv"}
: ${log_path="$root_dir/logs/client/${my_id}"}
: ${run_mode="RELEASE"}

export PYTHONPATH=${root_dir}
export LD_LIBRARY_PATH="$HOME/.local/lib/"

[ -e ${log_path} ] || mkdir -p ${log_path}

python3 ${root_dir}/src/client/main.py \
  --slo ${slo} \
  --frame_rate ${fps} \
  --init_bw 100000 \
  --model_server_host ${model_server_host} \
  --stats_fname ${stats_fname} \
  --run_mode "${run_mode}" \
  --video_file "${video_file}" \
  --image_list ${image_file} \
  --log_path "${log_path}" \
  > "${log_path}/stdout.log" 2>&1
