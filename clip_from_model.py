#!/usr/bin/python3


# See https://huggingface.co/docs/transformers/model_doc/clip
#   and CLIPTokenizer?

# Alternatively, consider what ComfyUI does? 
# See https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd1_clip.py
# with config of comfy/sd1_clip_config.json
#  model_class=CLIPTextModel
#   transformer=model_class.from_pretrained(/model/path)

import os
from transformers import CLIPTextModel

modelpath=os.path.join(os.path.dirname(os.path.realpath(__file__)), "AnythingV5Ink_ink.safetensors")


processor=None

def init():
    global processor
    # Load the processor and model
    print("loading "+modelpath)
    processor = CLIPTextModel.from_pretrained(modelpath,from_tf=True,local_files_only=True)
    print("done")

def tokenize_text(text):
    tokens = processor(text, return_tensors="pt")
    iid = tokens["input_ids"]
    # iid is expected to look like
    #   tensor([[49406,   320,  3638,  2677, 49407]])

    print(text , "= " , iid[0,1].item())

def tokenize_word(text):
    tokens = processor(text, return_tensors="pt")
    iid = tokens["input_ids"]
    # iid is expected to look like
    #   tensor([[49406,   320, 49407]])
    tensor1= iid[0]
    print(text , "= " , tensor1)
    print(text , "= " , tensor1[1:-1].tolist())


init()
sys.exit(0)

#input_text = "A Large Tree"
#input_text = "Large"
input_text = "unworthier"
tokenize_text(input_text)
tokenize_word(input_text)


# unworthier =  tensor([[49406,   569,   641, 25595, 49407]])

