# tokenspace directory

This directory contains utilities for the purpose of browsing the
"token space" of CLIP ViT-L/14

Long term goal is to be able to literally browse from word to 
"nearby" word



## generate-embeddings.py

Generates the "embeddings.safetensor" file!

Basically goes through the fullword.json file, and 
generates a standalone embedding object for each word.
Shape of the embeddings tensor, is
 [number-of-words][768]

## fullword.json

This file contains a collection of "one word, one CLIP token id" pairings.
The file was taken from vocab.sdturbo.json
First all the non-</w> entries were stripped out.
Then all the garbage punctuation and foreign characters were stripped out.
Finally, the actual </w> was stripped out, for ease of use.
