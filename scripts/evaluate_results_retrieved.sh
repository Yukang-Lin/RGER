#!/bin/bash
task=$1
for model_name in "llama2-7b-chat" "vicuna-7b"; do
    for method in "EPR" "CEIL" "DQ-LoRe" "RGER"; do
        bs_dir="output/${task}/${task}/${model_name}/${method}"
        files=(${bs_dir}/pred_*.json)

        if [ ${#files[@]} -eq 0 ]; then
            echo "no such file"
        else
            echo "processing ${files}"
            bname=$(basename ${files})
            output_file="results/${task}/${bname}"
            python evaluate_results.py --task ${task} --response_file ${files} --output_file ${output_file}
            
        fi
    done
done