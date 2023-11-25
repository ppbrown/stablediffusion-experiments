#!/usr/bin/python3

# Similar to clip.token.py, but
# This one uses the "pip clip" module, instead
# of the transformers thingie.
# It caches under ~/.cache/clip/
# So, the clip specifiation is different, even though
# in theory we get the same clip model, and same results

# https://colab.research.google.com/github/sagiodev/stablediffusion_webui/blob/master/Stable_Diffusion_tokenizer_and_embedding_SDA.ipynb

import clip
import torch
import sys

# To see other clip model names for this, use
# clip.available_models()
clipmodel="ViT-L/14"

processor=None
model=None

def init():
    global processor
    global model

    print("loading "+clipmodel)
    model, processor = clip.load(clipmodel)
    model.cuda().eval()
    print("done")

def tokenize_text(text):
    print(text+":",file=sys.stderr)
    tokens = clip.tokenize(text)

    print(tokens)
    return

    iid = tokens["input_ids"]
    # iid is expected to look like
    #   tensor([[49406,   320,  3638,  2677, 49407]])

    print(text , "= " , iid)

# Maybe use later, after tokenize_text works
def tokenize_word(text):
    tokens = processor(text, return_tensors="pt")
    iid = tokens["input_ids"]
    # iid is expected to look like
    #   tensor([[49406,   320, 49407]])
    tensor1= iid[0]
    print(text , "= " , tensor1)
    print(text , "= " , tensor1[1:-1].tolist())


print("Skipping init",file=sys.stderr)
# We only need to use init if we actually use the model
#init()

#input_text = "A Large Tree"
#input_text = "Large"
input_text = "unworthier"
input_text = "unworthy"
tokenize_text(input_text)
#tokenize_word(input_text)


# unworthier =  tensor([[49406,   569,   641, 25595, 49407]])

