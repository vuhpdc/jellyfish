#!/bin/bash

: ${root_dir:="$HOME/jellyfish/"}

## Download data
function gd_download() {
    [ -e $2 ] || mkdir -p $2
    dst="$2/$3"
    echo "Downloading file ${dst}"
    python3 ${root_dir}/google_drive.py $1 "$dst"
    tar -xvf "$dst" -C $2 > /dev/null
    rm "$dst"
}

# Download datasets that contains MS-COCO images and DDS traffic videos
dst_dir="${root_dir}/"
file="datasets.tar.gz"
gd_download 1sH1if2-T9zIs2y0m2CEGTficdyhSEdPM "$dst_dir" ${file}

# Download pytorch_yolov4 models, profiles and ground truth
dst_dir="${root_dir}/"
file="pytorch_yolov4.tar.gz"
gd_download 1WM1dBsRcX1PvVQSfg768R4Ao5cx4tZnh "$dst_dir" ${file}
