#!/usr/bin/python3

""" Work in progress
Plan:
   Read in fullword.json for list of words and token
   Read in pre-calculated "proper" embedding for each token from safetensor file
   Generate a tensor array of distance for each token, to every other token/embedding
   Save it out
"""


import sys
import json
import torch
from safetensors import safe_open

embed_file="embeddings.safetensors"

device=torch.device("cuda")

print("read in words from json now",file=sys.stderr)
with open("fullword.json","r") as f:
    tokendict = json.load(f)

print("read in embeddingsnow",file=sys.stderr)


model = safe_open(embed_file,framework="pt",device="cuda")
embs=model.get_tensor("embeddings")


print("Shape of result = ",embs.shape)


print("calculate distances now")


