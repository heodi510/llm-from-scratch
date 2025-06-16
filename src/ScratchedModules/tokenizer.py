import os
import re
import tiktoken
from importlib.metadata import version


class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.word2idx = vocab
        self.idx2word = {idx: word for word, idx in vocab.items()}
        self.size= len(vocab)+1
            
    def encode(self, text):
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        words = [item for item in words if item.strip()]
        return [self.word2idx[word] for word in words] 
        
    def decode(self, ids):
        words=[self.idx2word[idx] for idx in ids]
        sentence=' '.join(words)
        sentence = re.sub(r'\s([,.?!"()\'])', r'\1', sentence)
        return sentence
        
    def __len__(self):
        return self.size
    
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.word2idx = vocab
        self.idx2word = {idx: word for word, idx in vocab.items()}
        self.size= len(vocab)
            
    def encode(self, text):
        words = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        words = [item for item in words if item.strip()]
        return [self.word2idx.get(word, self.size-1) for word in words]
        
    def decode(self, ids):
        words=[self.idx2word[idx] for idx in ids]
        sentence=' '.join(words)
        sentence = re.sub(r'\s([,.?!"()\'])', r'\1', sentence)
        return sentence
        
    def __len__(self):
        return self.size
    