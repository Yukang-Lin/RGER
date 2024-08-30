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
from grakel.kernels import WeisfeilerLehman, VertexHistogram, ShortestPath
from grakel import graph_from_networkx
from transformers import set_seed
from src.utils.dpp_map import fast_map_dpp, k_dpp_sampling
from src.utils.misc import parallel_run, partial
from src.models.biencoder import BiEncoder
import pickle
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)

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
        self.dataset_reader = hu.instantiate(cfg.dataset_reader)

        if cfg.rerank and (cfg.graphsim or cfg.sensim_graphsim):
            logger.info("in loading graph structure")
            self.train_graphdict, self.test_graphdict = nx_graph_dict(cfg.graph_path,cfg.qa_model_name)

        self.encode_model = SentenceTransformer(cfg.encode_model)

        self.output_file = cfg.output_file
        self.num_candidates = cfg.num_candidates
        self.num_ice = cfg.num_ice
        self.is_train = cfg.dataset_reader.dataset_split == "train"

        if os.path.exists(cfg.faiss_index):
            logger.info(f"Loading faiss index from {cfg.faiss_index}")
            self.index = faiss.read_index(cfg.faiss_index)
            with open(cfg.index_dict, 'rb') as f:
                self.index_dict = pickle.load(f)
        else:
            self.index, self.index_dict = self.create_sensim_index(cfg)

    def create_sensim_index(self, cfg):
        logger.info("Building faiss index...")
        index_reader = hu.instantiate(cfg.index_reader)

        res_list = self.forward(index_reader)
        id_list = np.array([res['metadata']['id'] for res in res_list])
        embed_list = np.stack([res['embed'] for res in res_list])
        print(embed_list.shape,type(embed_list))

        index = faiss.IndexIDMap(faiss.IndexFlatIP(384))
        index.add_with_ids(embed_list, id_list)
        index_dict = dict(zip(id_list, embed_list))
        with open(cfg.index_dict, 'wb') as f:
            pickle.dump(index_dict, f)
        faiss.write_index(index, cfg.faiss_index)
        logger.info(f"Saving faiss index to {cfg.faiss_index}, size {len(index_reader)}")
        return index,index_dict

    def forward(self, dataset_reader):
        # encode
        res_list = []
        text_list = [item['metadata']['text'] for item in dataset_reader]
        res = self.encode_model.encode(text_list)
        for i, item in enumerate(dataset_reader):
            res_list.append({"embed": res[i], "metadata": item['metadata']})
        return res_list

    def find(self):
        res_list = self.forward(self.dataset_reader)
        for res in res_list:
            res['entry'] = self.dataset_reader.dataset_wrapper[res['metadata']['id']]
        print(res_list[0]['entry'])
        num_ice = self.cfg.rerank_ice if self.rerank else self.num_ice
        print('retrieve num_ice:',num_ice)
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

        postdata = []
        for d in data:
            entry = d['entry']
            entry['ctxs'] = d['ctxs']
            entry['ctxs_candidates'] = d['ctxs_candidates']
            postdata.append(entry)
        
        with open(self.output_file, "w") as f:
            json.dump(postdata, f)
        logger.info("save to {}".format(self.output_file))

    def graphsim_rerank(self,data):
        
        for id,entry in enumerate(tqdm.tqdm(data)):
            nx_query=self.test_graphdict[id][0]
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
            G=graph_from_networkx(nx_candidates,node_labels_tag='label')
            # kernel=ShortestPath(with_labels=True,normalize=True)
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
    
def nx_graph_dict(pth,model_name):
    # return [dict]
    # key: id,
    # value: (graph, variable exist or not, connected or not)
     
    def build_nx_graph(a):
    # to do: adapt to each task
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
    

@hydra.main(config_path="configs", config_name="sensim_retriever")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    dense_retriever = DenseRetriever(cfg)
    dense_retriever.find()


if __name__ == "__main__":
    main()
