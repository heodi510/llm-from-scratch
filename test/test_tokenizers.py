"""
test_tokenizers.py
===================

Compare different tokenizers with a sample text.3 tokenizer are tested:
1. SimpleTokenizerV1: Naive tokenizer that uses a simple mapping from text to ids.
2. SimpleTokenizerV2: Improved tokenizer that handles unknown words by mapping them to a special <unk> token.
3. tiktoken: A tokenizer that uses the GPT-2 encoding

Each tokenizer is tested with a sample text, and the results are printed to the console to show the difference in handling of known and unknown words.
For the first two tokenizers, the vocabulary is based on file loaded. Decoding an encoded text back to the original text back is not always possible if the text contains words not in the vocabulary.
For the tiktoken tokenizer, it can encode and decode any text without information loss even there are uncoomon words.

Usage:
    python3 test_tokenizers.py
"""

import tiktoken

from ScratchedModules.dataset import load_vocabs
from ScratchedModules.tokenizer import SimpleTokenizerV1, SimpleTokenizerV2


def test_tiktokenizer():
    print( "Testing Titoken....")
    tokenizer = tiktoken.get_encoding('gpt2')
    text = (
    "Hello, do you like tea <|endoftext|> In the sunlut terraces"
    "of someunknownPlace."
    )
    ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    print('Encoding normal text:')
    print(ids)
    print('Decoding ids back to text:')
    print(tokenizer.decode(ids))
    
    text="AKwirw ier"
    print(f'Encoding special text:  {text}')

    ids = tokenizer.encode(text)
    print(ids)
    print('Decoding ids back to text:')
    print(tokenizer.decode(ids))
    
def test_simpletokenizer_v1(pytestconfig):
    print( "Testing SimpleTokenizerV1....")
    project_root = pytestconfig.rootpath
    data_file    = project_root / "data" / "the-verdict.txt"
    text2ids=load_vocabs(path=data_file)
    tokenizer  = SimpleTokenizerV1(text2ids)
    text = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print('Encoding text included in voacb(text2ids):')
    print(ids)
    print('Decoding ids back to text:')
    print(tokenizer.decode(ids))
    
    text = " Hello, do you like the tea"
    print(f'Encoding text not included in voacb(text2ids):  {text}')
    try:
        ids = tokenizer.encode(text)
        print(ids)
        print('Decoding ids back to text:')
        print(tokenizer.decode(ids))
    except KeyError:
        # Handle missing words by mapping to the <unk> token index
        print(f"Contain word not found in vocabulary: {text}")
    print("===========================")

def test_simpletokenizer_v2(pytestconfig):
    print( "Testing SimpleTokenizerV2....")
    project_root = pytestconfig.rootpath
    data_file    = project_root / "data" / "the-verdict.txt"
    text2ids=load_vocabs(path=data_file)
    tokenizer  = SimpleTokenizerV2(text2ids)
    text = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print('Encoding text included in voacb(text2ids):')
    print(ids)
    print('Decoding ids back to text:')
    print(tokenizer.decode(ids))
    
    text = " Hello, do you like the tea"
    print(f'Encoding text not included in voacb(text2ids):  {text}')
    try:
        ids = tokenizer.encode(text)
        print(ids)
        print('Decoding ids back to text:')
        print(tokenizer.decode(ids))
    except KeyError:
        # Handle missing words by mapping to the <unk> token index
        print(f"Contain word not found in vocabulary: {text}")
    print("===========================")
