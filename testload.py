#!/usr/bin/python3

from transformers import CLIPProcessor, CLIPModel

#modelfile="openai/clip-vit-large-patch14"
#modelfile="clip-vit-large-patch14"
modelfile="clip-pbrown/"
#modelfile="pytorch_model.bin"
#modelfile="AnythingV5Ink_ink.safetensors"
#modelfile="anythingV3_fp16.ckpt"

tokenconfig="tokenizer_config.json"

processor=None

def init_model():
    print("loading "+modelfile)
    global processor
    processor = CLIPProcessor.from_pretrained(modelfile)
    print("done")

init_model()
