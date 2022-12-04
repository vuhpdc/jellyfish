#!/bin/bash

# Please provide the values of the following env variables through command line
: ${total_iter=3}
: ${start_iter=0}
: ${server_username:="username"}
: ${server_ip:="localhost"}
: ${server_ssh_port:=22}
: ${client_username:="username"}
: ${client_ip:="localhost"}
: ${client_ssh_port:=22}
: ${client_network_iface:="eth0"}
: ${network_trace_type:="synthetic_trace"}
: ${root_dir="${HOME}/jellyfish/"}
: ${log_path="${root_dir}/logs/experiment_manager/${network_trace_type}"}

# Some constant variables
FPS="15 25"
SLOs="75 100 150"
CLIENTS="1 2 4 8"
VIDEOS="dds/trafficcam_1.mp4 dds/trafficcam_2.mp4 dds/trafficcam_3.mp4"

# Prepare clients config files by replacing template values with the actual value
clients_cfg_dir="${root_dir}/src/experiment_manager/clients_cfg/"
[ -e "${clients_cfg_dir}/.temp/" ] ||  mkdir "${clients_cfg_dir}/.temp/"
cp ${clients_cfg_dir}/* ${clients_cfg_dir}/.temp/
find "${clients_cfg_dir}/.temp/" -type f -exec sed -i "s/client_ip/${client_ip}/g" {} \;
find "${clients_cfg_dir}/.temp/" -type f -exec sed -i "s/client_ssh_port/${client_ssh_port}/g" {} \;
find "${clients_cfg_dir}/.temp/" -type f -exec sed -i "s/client_network_iface/${client_network_iface}/g" {} \;
find "${clients_cfg_dir}/.temp/" -type f -exec sed -i "s/client_username/${client_username}/g" {} \;

# Run experiments for all possible combinations
for((iter = start_iter; iter < total_iter; iter++))
do
  for slo in ${SLOs}
  do
    for fps in ${FPS}
    do
      for clients in ${CLIENTS}
      do
        for video in ${VIDEOS}
        do
          echo "Iter: ${iter}, Running experiment ${clients}_${fps}_${slo} with network shaping"
          log_path="${log_path}" iter=${iter} expr_manager_root_dir=${root_dir} \
          ${root_dir}/src/experiment_manager/run.sh \
            ${server_ip} \
            ${server_username} \
            ${server_ssh_port} \
            ${clients_cfg_dir}/.temp/client_cfg_template_${clients}_${fps}_${slo}.txt \
            ${video} \
            ${network_trace_type}
        done
      done
    done
  done
done

# Delete the clients config as it may contain some sensitive info
rm -r "${clients_cfg_dir}/.temp/"

# Signal running process on server to terminate
# ssh -n ${server_username}@${server_ip} -p ${server_ssh_port} "pkill --signal 15 run.sh"
