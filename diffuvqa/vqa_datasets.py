# import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torch
import json
import psutil
import datasets
from datasets import Dataset as Dataset2
import sys
import os
from PIL import Image


def load_data_vqa(
    batch_size, 
    seq_len, 
    deterministic=False, 
    args=None, 
    model_emb=None,
    transform=None,
    split=None,
    loaded_vocab=None,
    loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    
       """

    print('#'*30, '\nLoading text data...')

    data, data_lst = get_corpus(args, seq_len, split=split, loaded_vocab=loaded_vocab)

    dataset = ImageTextDataset(args, transform=transform, text_datasets=data, data_lst=data_lst, data_args=args, model_emb=model_emb, split=split)

    if split != 'test':
        # sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            # sampler=sampler,
            shuffle=True,
            num_workers=4,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            drop_last=True,
            # sampler=sampler,
            shuffle=True,
            num_workers=4,
        )

    # return data_loader
    if loop and split == 'valid' :
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return data_loader

def infinite_loader(data_loader):
    while True:
        yield from data_loader

def helper_tokenize(sentence_lst, vocab_dict, seq_len, split):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    raw_datasets = Dataset2.from_dict(sentence_lst)

    print(raw_datasets)
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def tokenize_function(examples):
        input_id_q = vocab_dict.tokenizer(examples['question'], padding='max_length', max_length=seq_len, truncation=True)['input_ids']
        input_id_a = vocab_dict.tokenizer(examples['answer'], padding='max_length', max_length=seq_len, truncation=True)['input_ids']

        result_dict = {'input_id_q': input_id_q, 'input_id_a': input_id_a}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=8,
        remove_columns=['question', 'answer'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    print('### tokenized_datasets', tokenized_datasets)

    print('### tokenized_datasets...example', tokenized_datasets['input_id_q'][0], len(tokenized_datasets['input_id_a'][0]))
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_q'])):
            mask_zero = [0] * len(group_lst['input_id_q'][i])
            mask_one = [1] * len(group_lst['input_id_a'][i])
        
            lst.append(group_lst['input_id_q'][i] + group_lst['input_id_a'][i])

            mask.append(mask_zero + mask_one)
        group_lst['input_ids'] = lst
        
        group_lst['input_mask'] = mask
        return group_lst
    

    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        num_proc=1,
        desc=f"merge and mask",
    )
    

    raw_datasets = datasets.DatasetDict()
    raw_datasets[split] = tokenized_datasets
    print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
    return raw_datasets


def get_corpus(args, seq_len, split, loaded_vocab=None):
    
    print('#'*30, '\nLoading dataset {} from {}...'.format(args.dataset, args.data_dir))

    if args.dataset == 'ImageCLEFmed-MEDVQA-GI-2023-Development-Dataset' or args.dataset == 'Kvasir_VQA' or args.dataset == "pvqa" or args.dataset=="VQAMED2019":
        data_lst = {'question':[], 'answer': [], 'image_name': []}
    else:
        data_lst = {'question':[], 'answer': [], 'image_name': [], 'qid': [], 'img_id': []}

    if split == 'train':
        print('### Loading from the TRAIN set...')
        path = f'{args.data_dir}/train.jsonl'
    elif split == 'valid':
        print('### Loading from the VALID set...')
        path = f'{args.data_dir}/valid.jsonl'
    elif split == 'test':
        print('### Loading from the TEST set...')
        path = f'{args.data_dir}/test.jsonl'
    else:
        assert False, "invalid split for dataset"

    with open(path, 'r') as f_reader:
        for line in f_reader:
            content = json.loads(line)
            if args.dataset == 'VQAMED2019' or args.dataset == 'Med_RAD' or args.dataset == "pvqa" or args.dataset=="VQAMED2019":
                for key in ['question', 'answer', 'image_name']:
                    if key in content:
                        if isinstance(content[key], str):
                            data_lst[key].append(content[key].strip())
                        else:
                            data_lst[key].append(content[key])
            else:
                for key in ['question', 'answer', 'image_name', 'qid', 'img_id']:
                    if key in content:
                        if isinstance(content[key], str):
                            data_lst[key].append(content[key].strip())
                        else:
                            data_lst[key].append(content[key])

    print('### Data samples...\n', data_lst['question'][:2], data_lst['answer'][:2], data_lst['image_name'][:2])

    vocab_dict = loaded_vocab

    train_dataset = helper_tokenize(data_lst, vocab_dict, seq_len, split)
    return train_dataset, data_lst


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, data_lst, model_emb=None, split=None):
        super().__init__()  
        self.text_datasets = text_datasets  
        self.length = len(self.text_datasets[split])
        self.data_args = data_args  
        self.data_lst = data_lst
        self.model_emb = model_emb  
        self.split = split


    def __len__(self):
        return self.length  

    def __getitem__(self, idx):
        with (torch.no_grad()):  
            input_ids = self.text_datasets[self.split][idx]['input_ids']
            hidden_state = self.model_emb(torch.tensor(input_ids))


            arr = np.array(hidden_state, dtype=np.float32)


            out_kwargs = {}
            out_kwargs['input_q_id'] = np.array(self.text_datasets[self.split][idx]['input_id_q'])
            out_kwargs['input_a_id'] = np.array(self.text_datasets[self.split][idx]['input_id_a'])
            out_kwargs['input_ids'] = np.array(self.text_datasets[self.split][idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets[self.split][idx]['input_mask'])
            out_kwargs['image_name'] = self.data_lst['image_name'][idx]
            # out_kwargs['qid'] = self.data_lst['qid'][idx]
            # out_kwargs['img_id'] = self.data_lst['img_id'][idx]

            return arr, out_kwargs  

class ImageDataset(Dataset):
    def __init__(self, image_root, data_lst, args, transform=None):
        super().__init__()  
        self.image_root = image_root  
        self.data_lst = data_lst
        self.args = args  
        self.transform = transform
    
    def __len__(self):
        return len(self.data_lst['image_name'])

    def load_image_path(self):
        image_path = []
        for image_name in self.data_lst['image_name']:
            image_path.append(f'{self.image_root}/{image_name}')
        return image_path

    def __getitem__(self, idx):
        image_path = self.load_image_path()[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

class ImageTextDataset(Dataset):
    def __init__(self, args, transform=None, text_datasets=None, data_lst=None, data_args=None, model_emb=None, split=None):
        super().__init__()  
        self.image_dataset = ImageDataset(args.image_dir, data_lst, args, transform)   
        self.text_dataset = TextDataset(text_datasets, data_args, data_lst, model_emb, split)
        self.length = self.text_dataset.__len__()

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        image = self.image_dataset[idx]
        _, cond = self.text_dataset[idx]
        return image, cond

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):

    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result


if __name__ == "__main__":
    import sys
    import os
    from torchvision import transforms
    from transformers import BertTokenizer, BertModel
    import argparse
    import argparse
    import json, torch, os
    import numpy as np
    from diffuvqa.vqa_datasets import load_data_vqa
    from diffuvqa.step_sample import create_named_schedule_sampler
    from basic_utils import (
        load_model_emb,
        load_tokenizer)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/Med_RAD')
    parser.add_argument('--dataset', type=str, default='Med_RAD')
    parser.add_argument('--image_dir', type=str, default='datasets/Med_RAD/image_folder')
    parser.add_argument('--vocab_path', type=str, default='datasets/vocab.json')
    parser.add_argument('--vocab', type=str, default='bert')
    parser.add_argument('--config_name', type=str, default='bert-base-uncased')
    parser.add_argument('--checkpoint_path', type=str, default='diffuvqa/config')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--seq_len', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--json_path', type=str, default='datasets/Med_RAD/train.jsonl')
    parser.add_argument('--image_encoder', type=str, default='None')
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()


    tokenizer = load_tokenizer(args)
    model_emb, tokenizer = load_model_emb(args, tokenizer)
    # model_weight = torch.nn.Embedding(tokenizer.vocab_size, args.hidden_dim)
    # torch.nn.init.normal_(model_weight.weight, mean=0, std=0.02)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    data = load_data_vqa(batch_size=args.batch_size, seq_len=args.seq_len, args=args, model_emb=model_emb,
                            transform=transform, split='train', loaded_vocab=tokenizer)
    image, cond = next(data)

    print(image, image.shape)
    print(cond['input_q_id'], cond['input_q_id'].shape  )
    print(cond['input_a_id'], cond['input_a_id'].shape)
    print(cond['input_ids'], cond['input_ids'].shape)
    print(cond['input_mask'], cond['input_mask'].shape)
    # print(text
    # print(cond)

    # image_encoder = load_image_encoder(args)
    # image_embedding = image_encoder(image_batch)
    # # mask = torch.broadcast_to(cond['input_mask'].unsqueeze(dim=-1), x_0.shape)
    # print(image_embedding.shape)


