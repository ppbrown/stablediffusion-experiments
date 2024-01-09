#!/usr/bin/python3

# Does same thing as clip.dict.py, but
# just using different clip code

import sys
import clip
import torch
import numpy as np

# To see other clip model names for this, use
# clip.available_models()
clipmodel="ViT-L/14"

processor=None
model=None

def init():
    global processor
    global model

    print("loading "+clipmodel,file=sys.stderr)
    model, processor = clip.load(clipmodel)
    model.cuda().eval()
    print("done",file=sys.stderr)

def truncate_trailing_zeros(tensor):
    """Truncates trailing zeros from a 1D tensor efficiently using torch.nonzero."""
    non_zero_indices = torch.nonzero(tensor)
    max_idx = non_zero_indices.max()  # Find last non-zero index

    return tensor[:max_idx + 1]  # Slice the tensor to keep only relevant elements

def tokenize_text(text):
    #print("converting => "+text)

    tokens = clip.tokenize(text)
    # For some reason this lib zero pads, whereas
    # the other one does not

    trimmed_tensor= truncate_trailing_zeros(tokens[0])

    print(text,"= ",trimmed_tensor.tolist())

# Unlike with pyclip.token, we MUST call init
init()

print("Reading from stdin now...",file=sys.stderr)

for line in sys.stdin:
    input_text = line.rstrip()
    tokenize_text(input_text)
