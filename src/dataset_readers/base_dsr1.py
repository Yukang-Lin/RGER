import torch
import logging
from src.dataset_readers.dataset_wrappers import get_dataset_wrapper
from src.utils.tokenizer_util import get_tokenizer

logger = logging.getLogger(__name__)

class BaseDatasetReader(torch.utils.data.Dataset):

    def __init__(self, task_name, model_name, field, dataset_path=None, dataset_split=None, ds_size=None) -> None:
        print("!!!!!!!!!!!!!!!!!!!!!!!!base_dsr model_name:{}".format(model_name))
        self.tokenizer = get_tokenizer(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.init_dataset(task_name, field, dataset_path, dataset_split, ds_size)

    def init_dataset(self, task_name, field, dataset_path, dataset_split, ds_size, truncation=True):
        print("===task_name:{}".format(task_name))
        print("===dataset_split:{}".format(dataset_split))
        print('===dataset_path:{}'.format(dataset_path))
        with open(dataset_path, 'r') as f:
            content=f.read()
        dataset = eval(content)
        # self.encoded_dataset = encode_field(self.tokenizer, self.dataset_wrapper, field, truncation)
        for item in dataset:
            encoded = self.tokenizer.encode_plus(item['prompt'], truncation=truncation, return_tensors='pt')
            item['input_ids'] = encoded.input_ids[0]
            item['attention_mask'] = encoded.attention_mask[0]
            if 'options' in item.keys():
                metadata = {'question':item['question'],'prompt':item['prompt'],'options':item['options']}
                item.pop('options')
            else:
                metadata = {'question':item['question'],'prompt':item['prompt']}
            item.pop('question')
            item.pop('prompt')
                
            item['metadata'] = metadata
            
        self.encoded_dataset = dataset
        
    def __getitem__(self, index):
        return self.encoded_dataset[index]

    def __len__(self):
        return len(self.encoded_dataset)
