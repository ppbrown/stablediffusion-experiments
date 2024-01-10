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
wordlist = list(tokendict.keys())

print("read in embeddingsnow",file=sys.stderr)


model = safe_open(embed_file,framework="pt",device="cuda")
embs=model.get_tensor("embeddings")
embs.to(device)


print("Shape of loaded embeds =",embs.shape)
print("calculate distances now")

distances = torch.cdist(embs, embs, p=2)
print("distances shape is",distances.shape)

targetword="cat"
targetindex=wordlist.index(targetword)
print("index of cat is",targetindex)
targetdistances=distances[targetindex]

smallest_distances, smallest_indices = torch.topk(targetdistances, 5, largest=False)

smallest_distances=smallest_distances.tolist()
smallest_indices=smallest_indices.tolist()

print("The smallest distance values are",smallest_distances)
print("The smallest index values are",smallest_indices)

for t in smallest_indices:
    print(wordlist[t])



"""
import torch.nn.functional as F
pos=0
for word in tokendict.keys():
    print("Calculating distances from",word)
    home=embs[pos]
    #distances = torch.cdist(embs, home.unsqueeze(0), p=2)
    #distance = F.pairwise_distance(home, embs[,p=2).item()
"""
