import json
import logging
import faiss
import hydra
import hydra.utils as hu
import numpy as np
import torch
import tqdm
import os
import networkx as nx
import re
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath, RandomWalk
from grakel import graph_from_networkx
from transformers import set_seed
from torch.utils.data import DataLoader
from src.utils.dpp_map import fast_map_dpp, k_dpp_sampling
from src.utils.misc import parallel_run, partial
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.models.biencoder import BiEncoder
import pickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
logger = logging.getLogger(__name__)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Your code using the transformers library goes here
# import debugpy
# try:
#     debugpy.listen(("localhost", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print('error in debugpy')


class DenseRetriever:
    def __init__(self, cfg) -> None:
        self.cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.cfg=cfg
        self.rerank=cfg.rerank
        self.graphsim = cfg.graphsim
        if cfg.dq_file_path:
            cfg.dataset_reader.dataset_path = cfg.dq_file_path
            print("use dq_file_path as dataloader path")
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.cuda_device)
        self.dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        if cfg.rerank and (cfg.graphsim or cfg.sensim_graphsim):
            logger.info("in loading graph structure")
            self.train_graphdict, self.test_graphdict = nx_graph_dict(cfg.graph_path, cfg.task_name, cfg.qa_model_name)
        model_config = hu.instantiate(cfg.model_config)
        if cfg.pretrained_model_path is not None:
            self.model = BiEncoder.from_pretrained(cfg.pretrained_model_path, config=model_config)
        else:
            self.model = BiEncoder(model_config)

        self.model = self.model.to(self.cuda_device)
        self.model.eval()

        self.output_file = cfg.output_file
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.is_train = cfg.dataset_reader.dataset_split == "train"

        self.dpp_search = cfg.dpp_search
        self.dpp_topk = cfg.dpp_topk
        self.mode = cfg.mode

        if os.path.exists(cfg.faiss_index):
            logger.info(f"Loading faiss index from {cfg.faiss_index}")
            self.index = faiss.read_index(cfg.faiss_index)
            with open(cfg.index_dict, 'rb') as f:
                self.index_dict = pickle.load(f)
        else:
            self.index, self.index_dict = self.create_index(cfg)

    
    def create_index(self, cfg):
        logger.info("Building faiss index...")
        index_reader = hu.instantiate(cfg.index_reader)
        co = DataCollatorWithPaddingAndCuda(tokenizer=index_reader.tokenizer, device=self.cuda_device)
        dataloader = DataLoader(index_reader, batch_size=cfg.batch_size, collate_fn=co)

        res_list = self.forward(dataloader, encode_ctx=True)
        id_list = np.array([res['metadata']['id'] for res in res_list])
        embed_list = np.stack([res['embed'] for res in res_list])
        print(embed_list.shape,type(embed_list))

        index = faiss.IndexIDMap(faiss.IndexFlatIP(768))
        index.add_with_ids(embed_list, id_list)
        index_dict = dict(zip(id_list, embed_list))
        with open(cfg.index_dict, 'wb') as f:
            pickle.dump(index_dict, f)
        faiss.write_index(index, cfg.faiss_index)
        logger.info(f"Saving faiss index to {cfg.faiss_index}, size {len(index_reader)}")
        return index,index_dict

    def forward(self, dataloader, **kwargs):
        # encode
        res_list = []
        for i, entry in enumerate(tqdm.tqdm(dataloader)):
            with torch.no_grad():
                metadata = entry.pop("metadata")
                res = self.model.encode(**entry, **kwargs)
            res = res.cpu().detach().numpy()
            res_list.extend([{"embed": r, "metadata": m} for r, m in zip(res, metadata)])
        return res_list

    def find(self):
        res_list = self.forward(self.dataloader)
        for res in res_list:
            res['entry'] = self.dataset_reader.dataset_wrapper[res['metadata']['id']]
        print(res_list[0]['entry'])
        num_ice = self.cfg.rerank_ice if self.rerank else self.num_ice
        print('retrieve num_ice:',num_ice)
        if self.dpp_search:
            logger.info(f"Using scale_factor={self.model.scale_factor}; mode={self.mode}")
            func = partial(dpp, num_candidates=self.num_candidates, num_ice=num_ice,
                           mode=self.mode, dpp_topk=self.dpp_topk, scale_factor=self.model.scale_factor)
        else:
            func = partial(knn, num_candidates=self.num_candidates, num_ice=num_ice)
        data = parallel_run(func=func, args_list=res_list, initializer=set_global_object,
                            initargs=(self.index, self.is_train))
        
        if self.rerank:
            # rerank
            if self.cfg.pca:
                logger.info("in pca rerank")
                func = partial(self.pca_rerank,num_components=self.cfg.pca_dc)
            elif self.cfg.graphsim:
                logger.info("in graphsim rerank")
                func = partial(self.graphsim_rerank)
            elif self.cfg.sensim_graphsim:
                logger.info("in sensim_graphsim rerank")
                func = partial(self.sensim_graphsim_rerank,alpha=self.cfg.alpha)
            else:
                raise NotImplementedError
            # data = parallel_run(func=func, args_list=data)
            data = func(data)

        if self.dpp_search:
            postdata = data
        else:
            postdata = []
            for d in data:
                entry = d['entry']
                entry['ctxs'] = d['ctxs']
                entry['ctxs_candidates'] = d['ctxs_candidates']
                postdata.append(entry)
        
        with open(self.output_file, "w") as f:
            json.dump(postdata, f)
        logger.info("save to {}".format(self.output_file))

    def sensim_graphsim_rerank(self,data,alpha):
        
        for id,entry in enumerate(tqdm.tqdm(data)):
            
            candidate_ids=entry['ctxs']
            candict=dict(zip(range(len(entry['ctxs'])),entry['ctxs']))
            embed_list=[self.index_dict[id] for id in candidate_ids]
            embed_list=[entry['embed']]+embed_list
            embed=torch.Tensor(np.array(embed_list))
            embed_pca=PCA_svd(embed, self.cfg.pca_dc)
            query =embed_pca[0]
            embed_pca /= np.linalg.norm(embed_pca,axis=1,keepdims=True)
            # query = query[None,:]
            # query = np.ascontiguousarray(query).astype(np.float32)
            # rerank_array=np.ascontiguousarray(rerank_array).astype(np.float32)
            corr1 = np.matmul(query, embed_pca.T)
            corr1 = standardize(corr1[1:])
            # rerank_index = faiss.IndexIDMap(faiss.IndexFlatIP(self.cfg.pca_dc))
            # rerank_index.add_with_ids(rerank_array, np.array(candidate_ids,dtype=np.int64))
            # near_ids = rerank_index.search(query, self.cfg.num_ice)[1][0].tolist()

            nx_candidate=[self.train_graphdict[i][0] for i in candidate_ids]
            nx_candidate=[self.test_graphdict[id][0]]+nx_candidate
            # train graph not exist
            # test graph not exist
            G=graph_from_networkx(nx_candidate,node_labels_tag='label')
            # kernel=ShortestPath(with_labels=True,normalize=True)
            kernel = WeisfeilerLehman(n_iter=4,normalize=True,base_graph_kernel=VertexHistogram)
            corr2 = kernel.fit_transform(G)[0][1:]
            corr2 = standardize(corr2)
            corr = alpha*corr1 + (1-alpha)*corr2
            # corr = corr[1:]
            indices = np.argsort(corr).tolist()
            indices.reverse()
            entry['ctxs']=[candict[i] for i in indices[:self.cfg.num_ice]]
        return data

    def graphsim_rerank(self,data):
        
        for id,entry in enumerate(tqdm.tqdm(data)):
            nx_query=self.test_graphdict[id][0]
            entry['rerank_ice']=entry['ctxs']
            if nx_query is None:
                entry['ctxs']=entry['ctxs'][:self.cfg.num_ice]
                continue
            candidate_ids=entry['ctxs']
            candict,nx_candidates={},[nx_query]
            counter=1
            for i in candidate_ids:
                nx_candidate=self.train_graphdict[i][0]
                # contains three elements: graph, variable exist or not, connected or not
                # to do: if graph is not connected, delete it? for wl kernel, it is not necessary maybe
                if nx_candidate is None:
                    continue
                else:
                    candict[counter]=i
                    # for match graph index and candidate index
                    nx_candidates.append(nx_candidate)
                    counter+=1
            if self.cfg.task_name=='folio':
                G=graph_from_networkx(nx_candidates)
                kernel=ShortestPath(normalize=True,with_labels=False)
                
                # kernel = RandomWalk(normalize=True)
                # G = graph_from_networkx(nx_candidates)
                # G = graph_from_networkx(nx_candidates,node_labels_tag='label')
                # kernel = WeisfeilerLehman(n_iter=self.cfg.n_iter,normalize=True,base_graph_kernel=VertexHistogram)
            else:
                G=graph_from_networkx(nx_candidates,node_labels_tag='label')
                kernel = WeisfeilerLehman(n_iter=self.cfg.n_iter,normalize=True,base_graph_kernel=VertexHistogram)
            corr=kernel.fit_transform(G)
            indices=np.argsort(corr[0]).tolist()
            indices.reverse()
            indices=[i for i in indices[:self.cfg.num_ice+1] if i!=0]
            entry['ctxs']=[candict[i] for i in indices]
        return data

    def pca_rerank(self,data, num_components):
        # postdata=[]
        for entry in tqdm.tqdm(data):
            candidate_ids=entry['ctxs']
            # embed_list=[self.model.encode(self.index_reader[id]) for id in candidate_ids]
            embed_list=[self.index_dict[id] for id in candidate_ids]
            embed_list=[entry['embed']]+embed_list
            embed=torch.Tensor(np.array(embed_list))
            embed_pca=PCA_svd(embed, num_components)
            query,rerank_array=embed_pca[0],embed_pca[1:]
            query = query[None,:]
            query = np.ascontiguousarray(query).astype(np.float32)
            rerank_array=np.ascontiguousarray(rerank_array).astype(np.float32)
            rerank_index = faiss.IndexIDMap(faiss.IndexFlatIP(num_components))
            rerank_index.add_with_ids(rerank_array, np.array(candidate_ids,dtype=np.int64))
            near_ids = rerank_index.search(query, self.cfg.num_ice)[1][0].tolist()
            entry['ctxs']=near_ids
            # ctxs_candidates ???
            entry['ctxs_candidates']=[[i] for i in near_ids[:self.num_candidates]]
        return data

def standardize(embed):
    if isinstance(embed,list):
        embed=np.array(embed)
    return (embed - embed.mean()) / embed.std()

def extend_array_with_zeros(arr, target_length):
    if len(arr) >= target_length:
        return arr[:target_length]

    extended_arr = arr + [0] * (target_length - len(arr))
    return extended_arr

def high_op(ops):
    for op in ops:
        if op in ['*','/','#','<','>','|','&','^']:
            return op
    else:
        return ops[0]
    
def nx_graph_dict(pth, task_name, model_name):
    # return [dict]
    # key: id,
    # value: (graph, variable exist or not, connected or not)
     
    def build_logic_graph(a):
        G=nx.DiGraph()
        G.add_nodes_from([i['id'] for i in a['nodes']])
        G.add_edges_from([(i['source'],i['target']) for i in a['links']])
        node_attributes={node['id']:node['value'][1:] for node in a['nodes']}
        nx.set_node_attributes(G,node_attributes,'label')

        num=nx.number_weakly_connected_components(G)
        connected=num==1
        var=False
        # for alignment
        return G,var,connected

    def build_math_graph(a):
        var=False
        G=nx.DiGraph()
        G.add_nodes_from([i['id'] for i in a['nodes']])
        G.add_edges_from([(i['source'],i['target']) for i in a['links']])
        node_attributes={}
        for node in a['nodes']:
            if node['op']==[]:
                # node_attributes[node['id']]=node['value']
                if node['value']=='@@answer@@':
                    node_attributes[node['id']]='ans'
                elif re.search(r'[a-zA-Z]',node['value']):
                    node_attributes[node['id']]='var'
                    var=True
                # elif '%' in node['value']:
                #     node_attributes[node['id']]='%'
                else:
                    node_attributes[node['id']]='num'
            else:
                node_attributes[node['id']]=high_op(node['op'])
        nx.set_node_attributes(G,node_attributes,'label')
        num=nx.number_weakly_connected_components(G)
        connected=num==1
        return G,var,connected
    
    if task_name=='folio':
        build_nx_graph=partial(build_logic_graph)
    else:
        build_nx_graph=partial(build_math_graph)

    traindict,testdict={},{}
    train_dir=os.path.join(pth, 'train')
    print(f'loading train set from {train_dir} path...')
    for i in range(len(os.listdir(train_dir))):
        with open(f'{pth}/train/{i}.json','r') as f:
            a=json.load(f)
        if a is None:
            traindict[i]=(None,False,False)
        else:
            nxgraph,var,connected=build_nx_graph(a)
            traindict[i]=(nxgraph,var,connected)
    test_dir=os.path.join(pth, 'test', model_name)
    print(f'loading test set from {test_dir} path...')
    for i in range(len(os.listdir(test_dir))):
        with open(f'{pth}/test/{model_name}/{i}.json','r') as f:
            a=json.load(f)
        if a is None:
            testdict[i]=(None,False,False)
        else:
            nxgraph,var,connected=build_nx_graph(a)
            testdict[i]=(nxgraph,var,connected)
    print('done!')
    return traindict,testdict

def PCA_svd(X, k, center=True):
    # X = X.to(device)
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    # H = H.to(device)
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:k].t()
    
    return components.detach().cpu().numpy()


def set_global_object(index, is_train):
    global index_global, is_train_global
    index_global = index
    is_train_global = is_train


def knn(entry, num_candidates=1, num_ice=1):
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, max(num_candidates, num_ice)+1)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids

    # entry = entry['entry']
    entry['ctxs'] = near_ids[:num_ice]
    entry['ctxs_candidates'] = [[i] for i in near_ids[:num_candidates]]
    return entry
    
def long_knn(entry, num_candidates=1, num_ice=1):
    #print("######################### entry:{}".format(entry))
    embed = np.expand_dims(entry['embed'], axis=0)
    near_ids = index_global.search(embed, 16)[1][0].tolist()
    near_ids = near_ids[1:] if is_train_global else near_ids
    
    answer_len = len(entry['metadata']['answer'].split("\n"))
    #print("######################### entry['metadata']['answer']:{}".format(entry['metadata']['answer']))
    #print("######################### answer_len:{}".format(answer_len))
    
    with open('./index_data/gsm8k/index_dataset.json', 'r', encoding='latin-1') as file:
        data = json.load(file)
    
    #print("######################### data[0]['answer']:{}".format(data[0]['answer']))
    lenth = [len(data[i]['answer'].split("\n")) for i in near_ids]    

    #print("######################### answer_len:{}, lenth:{}, len_score:{}".format(answer_len, lenth, len_score))
    #print("######################### len_score:{}".format(len_score))
    sorted_ids = [x for _, x in sorted(zip(lenth, near_ids), reverse=True)]
    #print("######################### sorted_ids:{}".format(sorted_ids))
    
    #assert 1==0
    entry = entry['entry']
    entry['ctxs'] = sorted_ids[:num_ice]

    entry['ctxs_candidates'] = [[i] for i in sorted_ids[:num_candidates]]

    return entry


def get_kernel(embed, candidates, scale_factor):
    near_reps = np.stack([index_global.index.reconstruct(i) for i in candidates], axis=0)
    # normalize first
    embed = embed / np.linalg.norm(embed)
    near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=1)

    rel_scores = np.matmul(embed, near_reps.T)[0]
    # to make kernel-matrix non-negative
    rel_scores = (rel_scores + 1) / 2
    # to prevent overflow error
    rel_scores -= rel_scores.max()
    # to balance relevance and diversity
    rel_scores = np.exp(rel_scores / (2 * scale_factor))
    sim_matrix = np.matmul(near_reps, near_reps.T)
    # to make kernel-matrix non-negative
    sim_matrix = (sim_matrix + 1) / 2
    kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
    return near_reps, rel_scores, kernel_matrix


def random_sampling(num_total, num_ice, num_candidates, pre_results=None):
    ctxs_candidates_idx = [] if pre_results is None else pre_results
    while len(ctxs_candidates_idx) < num_candidates:
        # ordered by sim score
        samples_ids = np.random.choice(num_total, num_ice, replace=False).tolist()
        samples_ids = sorted(samples_ids)
        if samples_ids not in ctxs_candidates_idx:
            ctxs_candidates_idx.append(samples_ids)
    return ctxs_candidates_idx


def dpp(entry, num_candidates=1, num_ice=1, mode="map", dpp_topk=100, scale_factor=0.1):
    candidates = knn(entry, num_ice=dpp_topk)['ctxs']
    embed = np.expand_dims(entry['embed'], axis=0)
    near_reps, rel_scores, kernel_matrix = get_kernel(embed, candidates, scale_factor)

    if mode == "cand_random" or np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
        if np.isinf(kernel_matrix).any() or np.isnan(kernel_matrix).any():
            logging.info("Inf or NaN detected in Kernal_matrix, using random sampling instead!")
        topk_results = list(range(num_ice))
        ctxs_candidates_idx = [topk_results]
        ctxs_candidates_idx = random_sampling(num_total=dpp_topk,  num_ice=num_ice,
                                              num_candidates=num_candidates,
                                              pre_results=ctxs_candidates_idx)
    elif mode == "pure_random":
        ctxs_candidates_idx = [candidates[:num_ice]]
        ctxs_candidates_idx = random_sampling(num_total=index_global.ntotal,  num_ice=num_ice,
                                              num_candidates=num_candidates,
                                              pre_results=ctxs_candidates_idx)
        entry = entry['entry']
        entry['ctxs'] = ctxs_candidates_idx[0]
        entry['ctxs_candidates'] = ctxs_candidates_idx
        return entry
    elif mode == "cand_k_dpp":
        topk_results = list(range(num_ice))
        ctxs_candidates_idx = [topk_results]
        ctxs_candidates_idx = k_dpp_sampling(kernel_matrix=kernel_matrix, rel_scores=rel_scores,
                                             num_ice=num_ice, num_candidates=num_candidates,
                                             pre_results=ctxs_candidates_idx)
    else:
        # MAP inference
        map_results = fast_map_dpp(kernel_matrix, num_ice)
        map_results = sorted(map_results)
        ctxs_candidates_idx = [map_results]

    ctxs_candidates = []
    for ctxs_idx in ctxs_candidates_idx:
        ctxs_candidates.append([candidates[i] for i in ctxs_idx])
    # assert len(ctxs_candidates) == num_candidates

    entry = entry['entry']
    entry['ctxs'] = ctxs_candidates[0]
    entry['ctxs_candidates'] = ctxs_candidates
    return entry


@hydra.main(config_path="configs", config_name="dense_retriever")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    dense_retriever = DenseRetriever(cfg)
    dense_retriever.find()


if __name__ == "__main__":
    main()
