#!/bin/bash

: ${root_dir="${HOME}/jellyfish/"}
: ${my_dir="${root_dir}/plots/fig_8/"}
: ${synthetic_trace_logs="${root_dir}/logs/experiment_manager/synthetic_trace/network_shaping/"}
: ${total_iter:=3}

[ -e "${my_dir}/data" ] || mkdir ${my_dir}/data

export PYTHONPATH=${root_dir}

# Generate data in a format recognized by the plotting script
python3 ${my_dir}/generate_data.py \
    --data_dir ${synthetic_trace_logs} \
    --output_dir ${my_dir}/data \
    --num_iter ${total_iter} \
    > /dev/null 2>&1


# Plot the graph
python3 ${my_dir}/acc_miss_rate.py \
    --data ${my_dir}/data \
    --output_dir ${my_dir}

# Remove the temporary data directory
rm -r ${my_dir}/data