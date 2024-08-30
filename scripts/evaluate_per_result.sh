#!/bin/bash
task=$1
response_file=YOUR_OUTPUT_FILE_PATH
file_name=$(basename ${response_file})
echo file_name: ${file_name}
output_file=results/${task}/${file_name}
python evaluate_results.py --task ${task} --response_file ${response_file} --output_file ${output_file}
