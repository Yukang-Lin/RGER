task_name=$1
num_cluster=$2
encoder="/new_disk/med_group/lyk/models/all-MiniLM-L6-v2"

python autocot_retriever.py \
--encoder_pth ${encoder} \
--task_name ${task_name} \
--num_cluster ${num_cluster} \
--output_pth cot-prompt/${task_name}_autocot.txt