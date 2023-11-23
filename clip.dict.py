#!/usr/bin/python3

import sys
from transformers import CLIPProcessor

clipsrc="openai/clip-vit-large-patch14"

processor=None

def init():
    global processor
    # Load the processor and model
    print("loading "+clipsrc,file=sys.stderr)
    processor = CLIPProcessor.from_pretrained(clipsrc)
    print("done",file=sys.stderr)

# Tokenize a full sentence
def tokenize_text(text):
    tokens = processor(text, return_tensors="pt")
    tokens = tokens["input_ids"]
    print(text , "= " , tokens)

# Act like you only care about the first word,
# even if you pass in multiple
def tokenize_word(text):
    tokens = processor(text, return_tensors="pt")
    iid = tokens["input_ids"]
    # iid is expected to look like
    #   tensor([[49406,   320, 49407]])
    # The first and last numbers are throwaway
    tensor1= iid[0]
    #print(text , "= " , tensor1)
    print(text , "= " , tensor1[1:-1].tolist())



init()

print("Reading from stdin now...",file=sys.stderr)

for line in sys.stdin:
    input_text = line.rstrip()
    #tokenize_text(input_text)
    tokenize_word(input_text)

