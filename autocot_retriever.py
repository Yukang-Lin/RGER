
import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import json
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--task_name", type=str, default="gsm8k")
argparser.add_argument("--encoder_pth",type=str,default="/new_disk/med_group/lyk/models/all-MiniLM-L6-v2")
argparser.add_argument("--num_clusters",type=int,default=8)
argparser.add_argument("--random_seed",type=int,default=41)
argparser.add_argument("--sampling", type=str, default="center", help="whether to sample the cluster center first")
argparser.add_argument("--max_ra_step", type=int, default=5, help="maximum number of reasoning chains")
argparser.add_argument("--output_pth",type=str,default="cot-prompt/gsm8k_autocot.json")

def read_json(pth):
    with open(pth, 'r') as f:
        return json.load(f)
def save_json(data, pth):
    with open(pth, 'w') as f:
        json.dump(data, f)
def save_txt(data,pth):
    with open(pth, 'w') as f:
        f.write(data)

def process_data(data,task_name):
    if task_name=='gsm8k':
        for item in data:
            tmp=item['answer'].split('#### ')
            rationale,ans=tmp[0],tmp[1]
            item['rationale']=rationale+'The answer is '+ans+'.'
        corpus=[d['question'] for d in data]
    if task_name=='aqua':
        corpus=[d['question'] for d in data]
    if task_name=='svamp':
        for item in data:
            body,question=item.pop('Body'),item.pop('Question')
            item['question']=body+' '+question
            rationale=item.pop('Rationale1')
            # rationale='\n'.join(rationale.split('\n')[1:])
            rationale=rationale.strip('\n')
            item['rationale']=rationale.strip()
        corpus=[d['question'] for d in data]
    if task_name=='asdiv':
        for item in data:
            item['question']=item['Body']+' '+item['Question']
            item['rationale']=item['Rationale'].strip()
        corpus=[d['question'] for d in data]
    if task_name=='folio':
        for item in data:
            item['rationale']=item['rational'].strip()
        corpus=[d['question'] for d in data]
    return data, corpus

def main():
    args = argparser.parse_args()
    task_name = args.task_name
    num_clusters = args.num_clusters
    encoder = SentenceTransformer(args.encoder_pth)
    data_pth=f'index_data/{task_name}/index_dataset.json'
    data = read_json(data_pth)
    data,corpus = process_data(data,task_name)
    corpus_embeddings = encoder.encode(corpus)
    # shape len(corpus) * 384

    # Perform kmean clustering
    clustering_model = KMeans(n_clusters=num_clusters, random_state=args.random_seed)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    clustered_sentences = [[] for i in range(num_clusters)]
    dist = clustering_model.transform(corpus_embeddings)
    clustered_dists = [[] for i in range(num_clusters)]
    clustered_idx = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(corpus[sentence_id])
        clustered_dists[cluster_id].append(dist[sentence_id][cluster_id])
        clustered_idx[cluster_id].append(sentence_id)
    # select the closest sentence in each cluster
    demos=[]
    for i in range(len(clustered_dists)):
        print("Cluster ", i+1)
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        
        if not args.sampling == "center":
            random.shuffle(top_min_dist)
        for element in top_min_dist:
            min_idx = element[0]
            question = data[clustered_idx[i][min_idx]]['question'].strip()
            rationale = data[clustered_idx[i][min_idx]]['rationale'].strip()
            if task_name=='folio':
                demos.append({
                        "question": question,
                        "rationale": rationale
                    })
                break
            elif len(question.split()) <= 60 and len(rationale.split("\n")) <= args.max_ra_step:
                if task_name=='aqua':
                    demos.append({
                        "question": question,
                        "choices": data[clustered_idx[i][min_idx]]['options'],
                        "rationale": rationale
                    })
                else:
                    demos.append({
                        "question": question,
                        "rationale": rationale
                    })
                break
    # save the cot prompt
    auto_cot_prompt=''
    if task_name=='aqua':
        for demo in demos:
            auto_cot_prompt+="Q: "+demo['question']+'\n'+"Answer Choices: "+" ".join(demo['choices'])+'\nA:Let\'s think step by step\n'+demo['rationale']+'\n\n'
    else:
        for demo in demos:
            auto_cot_prompt+="Q: "+demo['question']+"\nA:Let's think step by step\n"+demo['rationale']+'\n\n'
    auto_cot_prompt=auto_cot_prompt[:-2]
    save_txt(auto_cot_prompt,args.output_pth)
    
if __name__ == "__main__":
    main()