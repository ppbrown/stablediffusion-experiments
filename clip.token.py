#!/usr/bin/python3

# This code works with the bare openai stuff.
# Alternatively, consider what ComfyUI does? 
# See https://github.com/comfyanonymous/ComfyUI/blob/master/comfy/sd1_clip.py
# with config of comfy/sd1_clip_config.json
#  model_class=CLIPTextModel
#   transformer=model_class.from_pretrained(/model/path)


from transformers import CLIPProcessor

# The common standard clip for models
#  (As seen in ghostmix/text_encoder/config.json, and others)
# is to use this as the standard... which gets passed to
# huggingface_hub module, which will then CACHE it under
#  ~/.cache/huggingface/hub/
clipsrc="openai/clip-vit-large-patch14"

# This does not work:
#clipsrc="stablediffusionapi/ghostmix"

processor=None

def init():
    global processor
    # Load the processor and model
    print("loading "+clipsrc)
    processor = CLIPProcessor.from_pretrained(clipsrc)
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
#input_text = "A Large Tree"
#input_text = "Large"
input_text = "unworthier"
tokenize_text(input_text)
tokenize_word(input_text)


# unworthier =  tensor([[49406,   569,   641, 25595, 49407]])

