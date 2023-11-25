#!/usr/bin/python3

# Takes pyclip.token.py further
# Does the same initial thing, then does the
# "embedding" thing, to generate the fullsize thing that
# gets passed to Unet?

import clip
import torch

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

def embed_text(text):
    print("converting => "+text)

    tokens = clip.tokenize(text)
    print("tokens: ",tokens)

    with torch.no_grad():
        embed = model.token_embedding(tokens.cuda())
    print("embed:", embed)
    print("embed shape:", embed.shape)


# Unlike with pyclip.token, we MUST call init
init()

#input_text = "unworthier"
input_text = "tree stump"
embed_text(input_text)


# unworthier =  tensor([[49406,   569,   641, 25595, 49407]])

