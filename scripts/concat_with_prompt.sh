#!/bin/bash
task_name=$1
prompt_type=$2
if [ "$prompt_type" == "complex_cot" ]; then
    python concat_with_prompt.py \
    --task_name ${task_name} \
    --complex True
    echo "complex-cot concat sucessfully!"
elif [ "$prompt_type" == "autocot" ]; then
    python concat_with_prompt.py \
    --task_name ${task_name} \
    --autocot True 
    echo "autocot concat sucessfully!"
elif [ ${prompt_type} == "cot" ]; then
    python concat_with_prompt.py \
    --task_name ${task_name}
    echo "cot concat sucessfully!"
else
    echo "Invalid prompt type"
fi