#!/bin/bash
# set -x
if [ $# -ne 7 ]; then
  echo "Usage: ./run_network_shaping.sh <server_host> <my_host_id> <slo> <fps> <video_name> <network_iface> <network_trace_type>"
  exit
fi

model_server_host=$1
my_host_id=$2
slo=$3
fps=$4
video_name=$5
my_rank=$( echo $my_host_id | awk -F'_' '{print $2}' )
network_iface=$6

: ${root_dir="$HOME/jellyfish/"}
: ${src_path="$root_dir/src/"}
: ${shaping_script="${root_dir}/network_shaping/shape_tbf.sh"}
: ${log_path:="${root_dir}/logs/client/${my_host_id}"}

# Network trace file and network trace interval
network_trace_file="${root_dir}/network_shaping/${7}.txt"
network_trace_interval=1
if [[ "$7" == "synthetic_trace" ]]; then
  network_trace_interval=20
fi

function stop() {
	kill -TERM ${python_client_pid}
	kill -TERM ${run_pid}
	kill -TERM ${shaping_pid}
	wait ${run_pid}
	wait ${shaping_pid}
	echo "Stopped!"
	exit
}

trap "stop" SIGHUP SIGINT SIGTERM

# Run client
${src_path}/client/run.sh ${model_server_host} ${my_host_id} ${slo} ${fps} ${video_name} &
run_pid="$!"
while true
do
	python_client_pid=$( pgrep -P $run_pid )
	if [ "$python_client_pid" != "" ]; then
		break
	fi
done
echo "$my_host_id: PIDs run:${run_pid}, python:${python_client_pid}"

# Get GRPC tcp port for python_client_pid
while true
do
	grpc_client_port="$( sudo netstat -apn  | grep "${python_client_pid}/python3" | awk -F' ' '{print $4}' | awk -F':' '{print $2}' )"
	if [ "${grpc_client_port}" != "" ]; then
		break
	fi
done
echo "$my_host_id: Client port:${grpc_client_port}"

# Start shaping script
# grpc_client_port=9999
shaping_log="${log_path}/network_shaping.csv"
shaping_log_err="${log_path}/network_shaping.err"
rm ${shaping_log}
sudo ${shaping_script} \
	${network_iface} \
	${network_trace_file} \
	${network_trace_interval} \
	${grpc_client_port} \
	${model_server_host} \
	10001 \
	${my_rank} \
	> ${shaping_log} 2> ${shaping_log_err} &
shaping_pid="$!"
echo "$my_host_id: Shaping pid: ${shaping_pid}"

# Wait for the client to finish
wait ${run_pid}
echo "${my_host_id}: Finished run.sh!"

# Signal shaping script to stop. Need to send signal to the child if sudo is used to execute the script.
for child in $( pgrep -P ${shaping_pid} );
do
	echo "Sending signal to $child"
	sudo kill -TERM ${child}
done
sudo kill -TERM ${shaping_pid}
wait ${shaping_pid}
