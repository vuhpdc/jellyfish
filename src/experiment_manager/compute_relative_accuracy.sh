#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: ./compute_relative_accuracy.sh <detections_dir> <gt_dir>"
    exit
fi

det_dir=$1
gt_dir=$2

: ${root_dir="${HOME}/jellyfish/"}
: ${output_det_file="output_dets.json"}
expr_manager_root_dir="${root_dir}/src/experiment_manager/"

# First convert output_det.json to per frame in text file
[ -e "${det_dir}/output_text_dir" ] || mkdir -p "${det_dir}/output_text_dir"
rm -r ${det_dir}/output_text_dir/*
python3 ${expr_manager_root_dir}/output_json2txt.py \
  --input_file "${det_dir}/${output_det_file}" \
  --output_dir "${det_dir}/output_text_dir"

# Now convert ground truth detections per frame in txt file format
[ -e "${gt_dir}/output_text_dir" ] || mkdir -p "${gt_dir}/output_text_dir"
python3 ${expr_manager_root_dir}/output_json2txt.py \
  --input_file "${gt_dir}/${output_det_file}" \
  --output_dir "${gt_dir}/output_text_dir" \
  --gt

## Now compute relative metric
echo "Make sure the objects that you want to detect is correctly set. HARDCODED!"

# F1 metric
python3 ${expr_manager_root_dir}/object_detection_metrics/eval_relative.py \
  --metric_type "F1" \
  -det "${det_dir}/output_text_dir" \
  -gt "${gt_dir}/output_text_dir" \
  | tee  "${det_dir}/accuracy_F1.txt"

# Clean up output directory of per frame text file
rm -r ${det_dir}/output_text_dir/*
