import os
import re
import torch
from torch.utils.data import Dataset, DataLoader

import tiktoken

verdict_path = os.path.join("..", "data", "the-verdict.txt")
def load_vocabs(path=verdict_path):
    """
    Load the text file which act as the base of the vocabulary.
    Default path is data/the-verdict.txt
    The file should be a plain text file with sentences.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print("Simple text preview:")
    print(raw_text[:99])
    preprocess = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocess = [item for item in preprocess if item.strip()]
    all_words =  sorted(set(preprocess))
    vocab_size= len(all_words)
    print(f"Vocab size:{vocab_size}")
    print ("Complete loading vocabulary from file:", path)
    print("==========================")
    text2ids={vocab: idx for idx, vocab in enumerate(all_words)}
    return text2ids


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids=[]
        self.target_ids=[]
        token_ids = tokenizer.encode(txt)
        
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk= token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self,idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, 
                         batch_size=4,
                         max_length=256,
                         stride=128,
                         shuffle=True,
                         drop_last=True,
                         num_workers=0):
    
    tokenizer=tiktoken.get_encoding('gpt2')
    dataset=GPTDatasetV1(txt, tokenizer,max_length, stride)
    data_loader=DataLoader(dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           drop_last=drop_last,
                           num_workers=num_workers
                           )
    return data_loader