#!/bin/bash
export WANDB_PROJECT=ICL  # change if needed
export WANDB_ENTITY=Your WANDB name  # change to your wandb account
export WANDB_API_KEY=Your API key  # change to your api-key
export WANDB_START_METHOD=thread
export TOKENIZERS_PARALLELISM=True
export HYDRA_FULL_ERROR=1

export CUDA_VISIBLE_DEVICES=0,2,3
# export CUDA_VISIBLE_DEVICES=$1
gpu=4
port=5324

task_name=$2
model_name=$3

if [ "${model_name}" == "llama2-7b-base" ]; then
  model_path=YOUR_PATH
elif [ "${model_name}" == "llama2-7b-chat" ]; then
  model_path=YOUR_PATH
elif [ "${model_name}" == "vicuna-7b" ]; then
  model_path=YOUR_PATH
elif [ "${model_name}" =="llama3-8b" ]; then 
  model_path=YOUR_PATH
fi
n_tokens=800
scr_batch_size=4
inf_batch_size=1


for task_name in ${task_name} # selection from asdiv gsm8k svamp aqua
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${task_name}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json   # If you wish to use a local dataset, please change this path to "index_data=index_data/${task_name}/${task_name}/train.json"
  mkdir -p ${run_dir}

  retrieve_file=${run_dir}/retrieved.json
  python bm25_retriever.py \
      hydra.run.dir=${run_dir}/bm25_retriever \
      output_file=${retrieve_file} \
      num_candidates=50 \
      num_ice=1 \
      task_name=${task_name} \
      index_reader.dataset_path=${index_data} \
      dataset_split=train \
      ds_size=44000 \
      query_field=a \
      index_reader.field=a

  scored_file=${run_dir}/scored.json
  accelerate launch --num_processes ${gpu} --main_process_port ${port}  scorer.py \
      hydra.run.dir=${run_dir}/scorer \
      model_name=${model_path} \
      task_name=${task_name} \
      output_file=${scored_file} \
      batch_size=${scr_batch_size} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data}

  run_name=bert-fix_ctx-shared-bs64_1
  run_dir=${run_dir}/${run_name}
  pretrained_model=${run_dir}/qa_model
  accelerate launch  --main_process_port ${port}  qa_retriever_trainer.py \
      hydra.run.dir=${run_dir}/trainer \
      task_name=${task_name} \
      qa_dataset_reader.dataset_path=${scored_file} \
      index_reader.dataset_path=${index_data} \
      training_args.output_dir=${run_dir} \
      training_args.run_name=${run_name} \
      model_config.ctx_model_name=null  \
      pretrained_model=${pretrained_model}
# share ctx model with q model 
done