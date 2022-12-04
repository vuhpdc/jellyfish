#!/bin/bash

: ${root_dir="${HOME}/jellyfish/"}
: ${my_dir="${root_dir}/plots/fig_6/"}

export PYTHONPATH=${root_dir}

python3 ${my_dir}/model_profiles.py \
    --data ${root_dir}/pytorch_yolov4/profiles/ \
    --output_dir ${my_dir}