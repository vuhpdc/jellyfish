#!/bin/bash
# set -x
# set -e
if [ $# -lt 5 ]; then
  echo "Usage: ./run.sh <server_host_ip> <server_user_name> <server_ssh_port> <clients_cfg_file> <video_name> [network_trace_type]"
  exit
fi

server_host=$1
server_user=$2
server_ssh_port=$3
clients_cfg_file=$4
video_name=$5

if [ $# -eq 6 ]; then
  network_trace_type=$6
fi

server_root_dir="/home/${server_user}/jellyfish/"

: ${iter:=0}
: ${expr_manager_root_dir="${HOME}/jellyfish/"}
: ${log_path="${expr_manager_root_dir}/logs/experiment_manager/"}
[ -e temp_clients_pid.txt ] && rm temp_clients_pid.txt

function get_filename() {
  fullfilename=$1
  filename=$(basename "$fullfilename")
  fname="${filename%.*}"
  return $fname
}

function stop() {
  echo "Stopping..."
  while IFS= read -r pid; do
    kill -TERM ${pid}
    wait $pid
    echo "Finshed pid ${pid}"
  done < temp_clients_pid.txt 
  rm temp_clients_pid.txt
  exit  
}

function random_wait() {
  rand_range_sec=$1
  wait_time_ms=$(($RANDOM % (${rand_range_sec} * 1000))) 
  wait_time_sec=$( echo "scale=3; ${wait_time_ms} / 1000" | bc )
  echo "Random wait of ${wait_time_sec} sec"
  sleep ${wait_time_sec}
}

trap "stop" SIGHUP SIGINT SIGTERM

# Start server first
# NOTE: Check if ~/.bashrc on the host has commented out to allow non-interactive ssh environment
# ssh -tt -n $server_user@$server_host -p $server_ssh_port "$server_root_dir/moth/server/run.sh" &
server_ssh_pid=$!
sleep 60 # Wait for enough time to start server processes.
echo "Server started!"

# Start all clients
while IFS=" " read -r host_id host port net_iface user root_dir slo fps
do
  # Wait for some random duration before we start the next client
  random_wait 10

  if [ -z "${network_trace_type}" ]; then
    echo "Running client ${host_id} on $user@$host without network shaping..."
    run_type="no_network_shaping"
    ssh -n $user@$host -p $port "$root_dir/src/client/run.sh ${server_host} ${host_id} ${slo} ${fps} ${video_name}" &
  else
    echo "Running client ${host_id} on $user@$host with network shaping..."
    run_type="network_shaping"
    ssh -n $user@$host -p $port "$root_dir/src/client/run_network_shaping.sh ${server_host} ${host_id} ${slo} ${fps} ${video_name} ${net_iface} ${network_trace_type}" &
  fi

  echo $! >> temp_clients_pid.txt
done < "${clients_cfg_file}"
echo "Started all clients"

# Wait for background shell commands to finish
while IFS= read -r pid; do
  wait $pid
  echo "Finshed pid ${pid}"
done < temp_clients_pid.txt 
echo "Finished all clients"
rm temp_clients_pid.txt

# Run cleanup and Copy log files locally
filename=$(basename "$clients_cfg_file")
experiment_type="${filename%.*}"
gt_model="1280_704"
video_name_wo_ext="${video_name%.*}"

# First Copy server logs
dst_dir="${log_path}/${run_type}/${experiment_type}/${video_name_wo_ext}/iter_${iter}/server"
[ -e ${dst_dir} ] || mkdir -p ${dst_dir}
scp -r -P ${server_ssh_port} ${server_user}@${server_host}:${server_root_dir}/logs/server/* ${dst_dir} > /dev/null
echo "Copied server logs to ${dst_dir}"

# Signal server to reset
ssh -n $server_user@$server_host -p $server_ssh_port "pkill --signal 2 run.sh"
ssh -n $server_user@$server_host -p $server_ssh_port "pkill --signal 2 run.sh"

# Now copy client logs
while IFS=" " read -r host_id host port net_iface user root_dir slo fps
do
  ssh -n $user@$host -p $port "$root_dir/src/client/run_cleanup.sh ${net_iface}"

  echo "Copying logs from client ${host_id} $user@$host..."
  dst_dir=${log_path}/${run_type}/${experiment_type}/${video_name_wo_ext}/iter_${iter}/${host_id}
  [ -e ${dst_dir} ] || mkdir -p ${dst_dir} 
  scp -r -P $port $user@$host:$root_dir/logs/client/${host_id}/* ${dst_dir} > /dev/null
  echo "Copied logs to ${dst_dir}"

  # Compute accuracy
  root_dir=${expr_manager_root_dir} ${expr_manager_root_dir}/src/experiment_manager/compute_relative_accuracy.sh ${dst_dir} ${expr_manager_root_dir}/pytorch_yolov4/ground_truth/${video_name_wo_ext}/model_${gt_model}/ > /dev/null
done < "${clients_cfg_file}"
