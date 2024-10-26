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
# choose from: asdiv, gsm8k, svamp, aqua, folio, prontoqa
method=$3
# EPR, CEIL, DQ-LoRe, RGER
model_name=$4
# vicuna-7b, llama2-7b-chat, llama3-8b
if [ "${task_name}" == "folio" ]; then
  num_ice=4
elif [ "${task_name}" == "prontoqa" ]; then
  num_ice=4
else
  num_ice=8
fi

port=5325

if [ "${model_name}" == "llama2-7b-base" ]; then
  model_path=/llama2-7b-base-hf
elif [ "${model_name}" == "llama2-7b-chat" ]; then
  model_path=/llama-2-7b-chat-hf
elif [ "${model_name}" == "vicuna-7b" ]; then
  model_path=/vicuna-7b
elif [ "${model_name}" == "llama3-8b" ]; then 
  model_path=/Meta-Llama-3-8B-Instruct
fi

n_tokens=800
inf_batch_size=1
scale_factor=0.1 # for CEIL

for task_name in ${task_name} 
do
  export WANDB_TAGS="${method},${task_name},${model_name}"
  run_dir=output/${task_name}/${task_name}/${model_name}
  index_data=index_data/${task_name}/index_dataset.json
  # If you wish to use a local dataset, please change this path to "index_data=index_data/${task_name}/${task_name}/train.json"
  mkdir -p ${run_dir}
  pretrained_model=${run_dir}/bert-fix_ctx-shared-bs64_2/qa_model
  run_dir=${run_dir}/${method}
  mkdir -p ${run_dir}

  if [ "${method}" = CEIL ]; then
    rerank_ice=100
  elif [ "${task_name}" == "gsm8k" ]; then
    rerank_ice=64
  else
    rerank_ice=64
  fi
  
  retrieve_file=${run_dir}/prompt-${method}_rerank${rerank_ice}_wl_test1.json
  echo "retrieve_file: ${retrieve_file}"
  pred_file=${run_dir}/pred_${method}_rerank${rerank_ice}_wl_test1.json
  
  dq_file_path=null
  if [ "${method}" = DQ-LoRe ]; then
    retrieve_field=dq_qa
    inference_field=dq_gen_a 
    dq_file_path=output/${task_name}/vanilla/${model_name}_complex_cot.json
  else
    retrieve_field=q
    inference_field=gen_a
  fi


  if [ ! -e ${retrieve_file} ]; then
    case "${method}" in
      "EPR")
        python dense_retriever.py \
          hydra.run.dir=${run_dir}/dense_retriever \
          output_file=${retrieve_file} \
          dataset_reader.dataset_path=./index_data/${task_name}/${task_name}/test.json \
          dataset_reader.field=${retrieve_field} \
          num_ice=${num_ice} \
          task_name=${task_name} \
          index_reader.dataset_path=${index_data} \
          pretrained_model_path=${pretrained_model} \
          faiss_index=${run_dir}/index \
          index_dict=${run_dir}/index_dict.pickle
        ;;
      "CEIL")
        python dense_retriever.py \
          hydra.run.dir=${run_dir}/dense_retriever \
          output_file=${retrieve_file} \
          dataset_reader.dataset_path=./index_data/${task_name}/${task_name}/test.json \
          dataset_reader.field=${retrieve_field} \
          num_ice=${num_ice} \
          task_name=${task_name} \
          index_reader.dataset_path=${index_data} \
          pretrained_model_path=${pretrained_model} \
          faiss_index=${run_dir}/index \
          index_dict=${run_dir}/index_dict.pickle \
          model_config.norm_embed=true \
          model_config.scale_factor=${scale_factor} \
          dpp_search=true \
          dpp_topk=${rerank_ice} \
          mode=map
        ;;
      "DQ-LoRe")
        python dense_retriever.py \
          hydra.run.dir=${run_dir}/dense_retriever \
          output_file=${retrieve_file} \
          dataset_reader.dataset_path=${dq_file_path} \
          dataset_reader.field=${retrieve_field} \
          num_ice=${num_ice} \
          task_name=${task_name} \
          index_reader.dataset_path=${index_data} \
          pretrained_model_path=${pretrained_model} \
          faiss_index=${run_dir}/index \
          index_dict=${run_dir}/index_dict.pickle \
          dq=True \
          dq_file_path=${dq_file_path} \
          rerank=true \
          rerank_ice=${rerank_ice} \
          pca=true
        ;;
      "RGER")
        python dense_retriever.py \
          hydra.run.dir=${run_dir}/dense_retriever \
          qa_model_name=${model_name} \
          output_file=${retrieve_file} \
          dataset_reader.dataset_path=./index_data/${task_name}/${task_name}/test.json \
          dataset_reader.field=${retrieve_field} \
          num_ice=${num_ice} \
          task_name=${task_name} \
          index_reader.dataset_path=${index_data} \
          pretrained_model_path=${pretrained_model} \
          faiss_index=${run_dir}/index \
          index_dict=${run_dir}/index_dict.pickle \
          rerank=true \
          rerank_ice=${rerank_ice} \
          graphsim=true \
          dq_file_path=${dq_file_path} \
          graph_path=save_graph/${task_name} \
          n_iter=3 
        ;;
    esac
  fi

  mkdir -p ${run_dir}/log
  nohup accelerate launch --num_processes ${gpu} --main_process_port ${port}  qa_inferencer.py \
      hydra.run.dir=${run_dir}/inferencer \
      model_config='hf-gen_a' \
      model_config.generation_kwargs.max_new_tokens=300 \
      model_name=${model_path} \
      task_name=${task_name} \
      dataset_reader.dataset_path=${retrieve_file} \
      dataset_reader.field=${inference_field} \
      dataset_reader.n_tokens=${n_tokens} \
      index_reader.dataset_path=${index_data} \
      output_file=${pred_file} \
      batch_size=${inf_batch_size} \
      log_path=${run_dir}/log > RGER_test1.out 2>&1 &
done