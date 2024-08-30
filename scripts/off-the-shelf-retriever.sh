#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
# export WANDB_ENTITY=Your WANDB name  # change to your wandb account
# export WANDB_API_KEY=Your API key  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=$1

gpu=1
task_name=$2
# choose from: asdiv, gsm8k, svamp, aqua
method=$3
# sensim, graphsim
num_ice=8
port=5324
model_name=$4
# vicuna-7b, llama2-7b-base, llama2-7b-chat

if [ "${model_name}" == "llama2-7b-base" ]; then
  model_path=YOUR_PATH
elif [ "${model_name}" == "llama2-7b-chat" ]; then
  model_path=YOUR_PATH
elif [ "${model_name}" == "vicuna-7b" ]; then
  model_path=YOUR_PATH
elif [ "${model_name}" == "llama3-8b-chat" ]; then
  model_path=YOUR_PATH
fi

n_tokens=800
scr_batch_size=4
inf_batch_size=1


rerank_ice=64

export WANDB_TAGS="${method},${task_name},${model_name}"
run_dir=output/${task_name}/${task_name}/${model_name}/${method}
index_data=index_data/${task_name}/index_dataset.json   # If you wish to use a local dataset, please change this path to "index_data=index_data/${task_name}/${task_name}/train.json"
mkdir -p ${run_dir}

retrieve_file=${run_dir}/prompt-${method}_rerank${rerank_ice}.json
if [ ! -e ${retrieve_file} ]; then
  if [ "${method}" == "sensim" ]; then
      python sensim_retriever.py \
          hydra.run.dir=${run_dir}/sensim_retriever \
          output_file=${retrieve_file} \
          dataset_reader.dataset_path=index_data/${task_name}/${task_name}/test.json \
          dataset_reader.field=q \
          num_ice=${num_ice} \
          task_name=${task_name} \
          index_reader.dataset_path=${index_data} \
          faiss_index=${run_dir}/sensim_index \
          index_dict=${run_dir}/sensim_index_dict.pickle
  else
      python sensim_retriever.py \
          hydra.run.dir=${run_dir}/sensim_retriever \
          output_file=${retrieve_file} \
          qa_model_name=${model_name} \
          dataset_reader.dataset_path=output/${task_name}/vanilla/${model_name}_complex_cot.json \
          num_ice=${num_ice} \
          task_name=${task_name} \
          index_reader.dataset_path=${index_data} \
          faiss_index=${run_dir}/sensim_index \
          index_dict=${run_dir}/sensim_index_dict.pickle \
          rerank=true \
          rerank_ice=${rerank_ice} \
          graphsim=true \
          graph_path=save_graph/${task_name} \
          n_iter=3
  fi
fi
mkdir -p ${run_dir}/log
pred_file=${run_dir}/pred_${method}_rerank_${rerank_ice}.json
accelerate launch --num_processes ${gpu} --main_process_port ${port}  qa_inferencer.py \
  hydra.run.dir=${run_dir}/inferencer \
  model_config='hf-gen_a' \
  model_name=${model_path} \
  task_name=${task_name} \
  dataset_reader.dataset_path=${retrieve_file} \
  dataset_reader.n_tokens=${n_tokens} \
  dataset_reader.field=gen_a \
  index_reader.dataset_path=${index_data} \
  output_file=${pred_file} \
  batch_size=${inf_batch_size} \
  log_path=${run_dir}/log