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

print("read in embeddings now",file=sys.stderr)

model = safe_open(embed_file,framework="pt",device="cuda")
embs=model.get_tensor("embeddings")
embs.to(device)
print("Shape of loaded embeds =",embs.shape)

# ("calculate distances now")
distances = torch.cdist(embs, embs, p=2)
print("distances shape is",distances.shape)

# Find 10 closest tokens to targetword.
# Will include the word itself
def find_closest(targetword):
    try:
        targetindex=wordlist.index(targetword)
    except ValueError:
        print(targetword,"not found")
        return

    #print("index of",targetword,"is",targetindex)
    targetdistances=distances[targetindex]

    smallest_distances, smallest_indices = torch.topk(targetdistances, 10, largest=False)

    smallest_distances=smallest_distances.tolist()
    smallest_indices=smallest_indices.tolist()
    for d,i in zip(smallest_distances,smallest_indices):
        print(wordlist[i],"(",d,")")
    #print("The smallest distance values are",smallest_distances)
    #print("The smallest index values are",smallest_indices)


print("Input a word now:")
for line in sys.stdin:
    input_text = line.rstrip()
    find_closest(input_text)
