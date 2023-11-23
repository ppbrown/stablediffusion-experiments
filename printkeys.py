#!/usr/bin/python3


import torch

import sys

if len(sys.argv) >1:
    filename=sys.argv[1]
else:
    filename="AnythingV5Ink_ink.safetensors"


print("loading "+filename)


model=torch.load(filename)


for key in model.keys():
    print(key)
