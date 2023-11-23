#!/usr/bin/python3


from safetensors import safe_open
#from safetensors.torch import safe_open

import sys

if len(sys.argv) >1:
    filename=sys.argv[1]
else:
    filename="AnythingV5Ink_ink.safetensors"


print("loading "+filename)


model=safe_open(filename,framework="pt")


for key in model.keys():
    print(key)
