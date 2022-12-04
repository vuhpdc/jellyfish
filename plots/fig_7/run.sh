#!/bin/bash

: ${root_dir="${HOME}/jellyfish/"}
: ${my_dir="${root_dir}/plots/fig_7/"}
: ${synthetic_trace_logs="${root_dir}/logs/experiment_manager/synthetic_trace/network_shaping/"}

[ -e "${my_dir}/data" ] || mkdir ${my_dir}/data

# Get the data from the logs needed for the plots
cp ${synthetic_trace_logs}/client_cfg_template_2_25_100/dds/trafficcam_1/iter_0/host_0/frame_stats.csv ${my_dir}/data/frame_stats_host_0.csv 
cp ${synthetic_trace_logs}/client_cfg_template_2_25_100/dds/trafficcam_1/iter_0/host_1/frame_stats.csv ${my_dir}/data/frame_stats_host_1.csv 
cp ${synthetic_trace_logs}/client_cfg_template_2_25_100/dds/trafficcam_1/iter_0/host_0/network_shaping.csv ${my_dir}/data/network_shaping_host_0.csv 
cp ${synthetic_trace_logs}/client_cfg_template_2_25_100/dds/trafficcam_1/iter_0/host_1/network_shaping.csv ${my_dir}/data/network_shaping_host_1.csv 
cp ${synthetic_trace_logs}/client_cfg_template_2_25_100/dds/trafficcam_1/iter_0/server/frame_path.csv ${my_dir}/data/frame_path_server.csv 

# Plot the graph
export PYTHONPATH=${root_dir}

python3 ${my_dir}/model_adaptation.py \
    --data ${my_dir}/data \
    --output_dir ${my_dir}

# Remove the temporary data directory
rm -r ${my_dir}/data