#!/usr/bin/python3


import torch

from safetensors import safe_open
#from safetensors.torch import open_file
#from safetensors.torch import safe_open

import safetensors
import sys

if len(sys.argv) >1:
    filename=sys.argv[1]
else:
    filename="AnythingV5Ink_ink.safetensors"


print("loading "+filename)


model=safe_open(filename,framework="pt")


for key in model.keys():
    print(key,model.get_slice(key))
    print("DEBUG: Exit now")
    sys.exit(0)
