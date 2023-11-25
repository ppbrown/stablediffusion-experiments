#!/usr/bin/python3

# See https://huggingface.co/docs/safetensors/index

from safetensors import safe_open

import sys

if len(sys.argv) >1:
    filename=sys.argv[1]
else:
    filename="AnythingV5Ink_ink.safetensors"


print("loading "+filename)


model=safe_open(filename,framework="pt")

# We want a sorted list of desired keys on stdin,
# so that it will be easier to compare outputs across files
for key in sys.stdin:
    key=key.rstrip()
    print(model.get_tensor(key))

