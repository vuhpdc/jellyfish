#!/bin/bash

: ${root_dir="${HOME}/jellyfish/"}
: ${my_dir="${root_dir}/plots/fig_9/"}
: ${synthetic_trace_logs="${root_dir}/logs/experiment_manager/synthetic_trace/network_shaping/"}
: ${total_iter:=3}

[ -e "${my_dir}/data" ] || mkdir ${my_dir}/data

# Function to get the data from the logs needed for the plots
generate_data() {
    FPS="15 25"
    SLOs="75 100 150"
    CLIENTS="1 2 4 8"
    VIDEOS="dds/trafficcam_1 dds/trafficcam_2 dds/trafficcam_3"
    for fps in $FPS
    do
        for slo in $SLOs
        do
            for clients in $CLIENTS
            do
                dst_file="${my_dir}/data/frame_stats_${clients}_${fps}_${slo}.csv"
                echo "" > ${dst_file}
                for (( host = 0; host < clients; host++))
                do
                    for video in ${VIDEOS}
                    do
                        for (( start_iter = 0; start_iter < total_iter; start_iter++))
                        do
                            src_file="${synthetic_trace_logs}/client_cfg_template_${clients}_${fps}_${slo}/${video}/iter_${start_iter}/host_${host}/frame_stats.csv"
                            sed 1d ${src_file} >> ${dst_file}
                            header=`head -1 ${src_file}`
                        done
                    done
                done

            # Copy header to the dst file
            sed -i "1 s/^/${header}/" -i ${dst_file}
            done
        done
    done
}

# Fetch data
generate_data

# Plot the graph
export PYTHONPATH=${root_dir}

python3 ${my_dir}/latency_cdf.py \
    --data ${my_dir}/data \
    --output_dir ${my_dir}

# Remove the temporary data directory
rm -r ${my_dir}/data
