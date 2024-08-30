import os
import random
import numpy as np
import json
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--task_name", type=str, default="gsm8k")
argparser.add_argument("--output_dir", type=str, default="output")
argparser.add_argument("--complex", type=bool, default=False)
argparser.add_argument("--autocot", type=bool, default=False)

def read_json(pth):
    with open(pth, 'r') as f:
        return json.load(f)
def save_json(data, pth):
    with open(pth, 'w') as f:
        json.dump(data, f)
def read_txt(pth):
    with open(pth, 'r') as f:
        return f.read()
def save_txt(data,pth):
    with open(pth, 'w') as f:
        f.write(data)

def concat_prompt(data,task_name,type,prompt):
    # return prompts, list of dict with keys 'question' and 'prompt'
    prompts=[]
    if type=='complex':
        qtype="Question"
    else:
        qtype="Q"
    if task_name=='gsm8k':
        if type=="cot":
            prompts=[{"question":d['question'] ,
                      "prompt":f"{prompt}\n\n{qtype}: {d['question']}\nA:"} for d in data]
        else:
            prompts=[{"question":d['question'], 
                      "prompt":f"{prompt}\n\n{qtype}: {d['question']}\nA: Let's think step by step\n"} for d in data]
    if task_name=='aqua':
        if type=='cot':
            prompts=[{
                "question":f"{d['question']}\nAnswer Choices: {' '.join(d['options'])}" , 
                "options":d['options'], 
                "prompt":f"{prompt}\n\n{qtype}: {d['question']}\nAnswer Choices: {' '.join(d['options'])}\nA:"} for d in data]
        else:
            prompts=[{
                "question":f"{d['question']}\nAnswer Choices: {' '.join(d['options'])}" , 
                "options":d['options'], 
                "prompt":f"{prompt}\n\n{qtype}: {d['question']}\nAnswer Choices: {' '.join(d['options'])}\nA: "} for d in data]
    if task_name=='svamp':
        if type=='cot':
            prompts=[{"question": d['Body']+' '+d['Question'],
                      "prompt":f"{prompt}\n\n{qtype}: {d['Body']} {d['Question']}\nA:"} for d in data]
        else:
            prompts=[{"question": d['Body']+' '+d['Question'],
                      "prompt":f"{prompt}\n\n{qtype}: {d['Body']} {d['Question']}\nA: Let's think step by step\n"} for d in data]
    if task_name=='asdiv':
        if type=='cot':
            prompts=[{"question":d['Body']+' '+d['Question'],
                      "prompt":f"{prompt}\n\n{qtype}: {d['Body']} {d['Question']}\nA:"} for d in data]
        else:
            prompts=[{"question":d['Body']+' '+d['Question'],
                      "prompt":f"{prompt}\n\n{qtype}: {d['Body']} {d['Question']}\nA: Let's think step by step\n"} for d in data]
    if task_name=='folio':
        if type=='cot':
            prompts=[{"question":d['Body']+' '+d['Question'],
                      "prompt":f"{prompt}\n\n{qtype}: {d['Body']} {d['Question']}\nA:"} for d in data]
        else:
            prompts=[{"question":d['question'],
                      "prompt":f"{prompt}\n\n{qtype}: {d['question']}\nA: Let's think step by step\n"} for d in data]
    return prompts

def main():
    args = argparser.parse_args()
    task_name = args.task_name
    # print(args)
    if args.complex:
        type="complex_cot"
    elif args.autocot:
        type="autocot"
    else:
        type="cot"
    # print(type)
    data_pth = f"index_data/{task_name}/{task_name}/test.json"
    data = read_json(data_pth)
    prompt_pth = f"cot-prompt/{task_name}_{type}.txt"
    prompt = read_txt(prompt_pth)
    prompts = concat_prompt(data,task_name,type,prompt)
    output_pth = f"{args.output_dir}/{task_name}/vanilla/{type}_prompt.json"
    save_json(prompts,output_pth)
    print("=====save in {}".format(output_pth))

if __name__ == "__main__":
    main()