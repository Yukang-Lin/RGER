import glob
import json
import os
import logging
import hydra
import hydra.utils as hu
import torch
import tqdm
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import set_seed
# from src.metrics import get_metric
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.utils.statistics import show_statistics
from src.models.api_client import run_api
from src.utils.misc import parallel_run, save_json
from src.models.model import ppl_generate
from datetime import datetime

import logging
import faiss
import numpy as np
import torch
import tqdm
import os
from transformers import set_seed
from torch.utils.data import DataLoader
from src.utils.dpp_map import fast_map_dpp, k_dpp_sampling
from src.utils.misc import parallel_run, partial
from src.utils.collators import DataCollatorWithPaddingAndCuda
from src.models.biencoder import BiEncoder

from transformers import BertTokenizer
from transformers import AutoTokenizer

import re
import itertools
# import debugpy
# try:
#     debugpy.listen(("localhost", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


logger = logging.getLogger(__name__)

class Inferencer:
    def __init__(self, cfg, accelerator=None) -> None:
        self.cuda_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset_reader = hu.instantiate(cfg.base_dataset_reader)
        self.cfg = cfg
        self.task_name = cfg.task_name
        self.accelerator = accelerator
        self.output_file = cfg.output_file
        # OmegaConf DictConfig to dict
        self.generation_kwargs = OmegaConf.to_object(cfg.model_config.generation_kwargs)
        self.model, self.dataloader = self.init_model_dataloader(cfg)

    def init_model_dataloader(self, cfg):
        co = DataCollatorWithPaddingAndCuda(tokenizer=self.dataset_reader.tokenizer, device=self.accelerator.device)
        dataloader = DataLoader(self.dataset_reader, batch_size=cfg.batch_size, collate_fn=co)

        model = hu.instantiate(cfg.model_config.model).eval()
        model = self.accelerator.prepare(model)

        if hasattr(model, "module"):
            model = model.module

        return model, dataloader

    def forward(self):
        if self.accelerator.is_main_process:
            dataloader = tqdm.tqdm(self.dataloader)
        else:
            dataloader = self.dataloader
        res = []
        for i, entry in enumerate(dataloader):
            metadata = entry.pop("metadata")
            if self.task_name == 'aqua':
                # for classification tasks, we compare the ppl of provided generation_choices as generation
                choices_list = [meta['options'] for meta in metadata]
                preds = ppl_generate([meta['prompt'] for meta in metadata],
                                     model=self.model,
                                     tokenizer=self.dataset_reader.tokenizer,
                                     choices_list=choices_list,
                                     device=self.accelerator.device)
                for mdata, pred in zip(metadata, preds):
                    mdata['generated'] = pred
            else:
                with torch.no_grad():
                    outputs = self.model.generate(input_ids=entry.input_ids,
                                                    attention_mask=entry.attention_mask,
                                                    eos_token_id=self.dataset_reader.tokenizer.eos_token_id,
                                                    pad_token_id=self.dataset_reader.tokenizer.pad_token_id,
                                                    # do_sample=False,  # always use greedy decode here
                                                    **self.generation_kwargs)
                    prompt_len = int(entry.attention_mask.shape[1])
                    # batch size must set to 1
                    if self.cfg.batch_size==1:
                        if self.generation_kwargs['num_return_sequences'] == 1:
                            generated = [self.dataset_reader.tokenizer.decode(outputs[0][prompt_len:])]
                        else:
                            generated = [self.dataset_reader.tokenizer.decode(output[prompt_len:]) for output in outputs.tolist()]    
                        generated = [g.strip(self.dataset_reader.tokenizer.pad_token).strip() for g in generated]
                        metadata[0]['generated'] = generated
                    else:
                        if self.generation_kwargs['num_return_sequences'] != 1:
                            raise ValueError("batch size must set to 1 when num_return_sequences > 1")
                        for mdata, output in zip(metadata, outputs.tolist()):
                            generated = self.dataset_reader.tokenizer.decode(output[prompt_len:])
                            mdata['generated'] = generated.strip(self.dataset_reader.tokenizer.pad_token).strip()
                        
            res.extend(metadata)

            if i == 0:
                logger.info(f"Prompt: {metadata[0]['prompt']}")
                logger.info(f"Generated: {metadata[0]['generated']}")

        save_json(self.output_file,res)
        # save_log 
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        save_json(f"{self.cfg.log_path}/{self.cfg.task_name}_{formatted_time}.json", res)
        # save in require path
        self.save_result(res)
    
    def save_result(self, res):
        with open(f'index_data/{self.cfg.task_name}/{self.cfg.task_name}/test.json','r') as f:
            backup = json.load(f)
        assert len(res)==len(backup)

        save=[]
        for i,item in enumerate(res):
            if self.cfg.task_name == 'aqua':
                question=backup[i]['question']+'\nAnswer Choices:'+' '.join(backup[i]['options'])
            elif self.cfg.task_name == 'svamp' or self.cfg.task_name == 'asdiv':
                question=backup[i]['Body'] + ' ' + backup[i]['Question']
            elif self.cfg.task_name == 'gsm8k':
                question=backup[i]['question']
            elif self.cfg.task_name == 'folio':
                question=backup[i]['premises'] + ' @@ ' + backup[i]['conclusion']
            else:
                raise NotImplementedError
            
            preds=item['generated']
            toans={0:'A',1:'B',2:'C',3:'D',4:'E'}
            if self.task_name == 'aqua':
                save.append({'question':question,'options':item['options'],'answer': toans[preds]})
            else:
                if len(preds)==1:
                    pred=preds[0].split('\n\n')[0]
                    save.append({'question':question,'answer': pred})
                else:
                    preds=[pred.split('\n\n')[0] for pred in preds]
                    save.append({'question':question,'answer':preds})
        save_json(self.output_file, save)
        print('save to:',self.output_file)
        return 
    
    
@hydra.main(config_path="configs", config_name="prompt_inferencer")
def main(cfg):
    logger.info(cfg)
    set_seed(43)
    accelerator = Accelerator()
    inferencer = Inferencer(cfg, accelerator)
    inferencer.forward()
    accelerator.wait_for_everyone()
    
if __name__ == "__main__":
    main()
