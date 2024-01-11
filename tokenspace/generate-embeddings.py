#!/usr/bin/python3

""" Work in progress
Plan:
   Read in fullword.json for list of works and token
   Generate "proper" embedding for each token, and store in tensor file
   Generate a tensor array of distance to every other token/embedding
   Save it out
"""


import sys
import json
import torch
from safetensors.torch import save_file
from transformers import CLIPProcessor,CLIPModel

clipsrc="openai/clip-vit-large-patch14"
processor=None
model=None

device=torch.device("cuda")


def init():
    global processor
    global model
    # Load the processor and model
    print("loading processor from "+clipsrc,file=sys.stderr)
    processor = CLIPProcessor.from_pretrained(clipsrc)
    print("done",file=sys.stderr)
    print("loading model from "+clipsrc,file=sys.stderr)
    model = CLIPModel.from_pretrained(clipsrc)
    print("done",file=sys.stderr)

    model = model.to(device)

# Expect SINGLE WORD ONLY
def standard_embed_calc(text):
    inputs = processor(text=text, return_tensors="pt")
    inputs.to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
    embedding = text_features[0]
    return embedding


init()

with open("dictionary","r") as f:
    tokendict = f.readlines()
    tokendict = [token.strip() for token in tokendict]  # Remove trailing newlines

print("generate embeddings for each now",file=sys.stderr)
count=1
all_embeddings = []
for word in tokendict:
    emb = standard_embed_calc(word)
    emb=emb.unsqueeze(0) # stupid matrix magic to make the cat work
    all_embeddings.append(emb)
    count+=1
    if (count %100) ==0:
        print(count)

embs = torch.cat(all_embeddings,dim=0)
print("Shape of result = ",embs.shape)
print("Saving all the things...")
save_file({"embeddings": embs}, "embeddings.safetensors")


print("calculate distances now")


