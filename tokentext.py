#!/usr/bin/python3

# my own test for tokenization by CLIP
# as suggested by gpt4

from transformers import CLIPProcessor, CLIPModel

#modelfile="clip-vit-base-patch32/"
#modelfile="./clip-pbrown"
modelfile="./anything-v5"
#modelfile="AnythingV5Ink_ink.safetensors"


processor=None

def init_model():
    # Load the processor and model
    print("loading "+modelfile)
    global processor
    processor = CLIPProcessor.from_pretrained(modelfile)
    print("done")

def tokenize_text(text):
    # Tokenize the input text
    tokens = processor(text, return_tensors="pt")

    # Print the input text and the resulting tokens
    print("Input Text:", text)
    print("Tokenization Result:", tokens)

    return

    """
    print("This should not get called!!")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    # Forward pass through the model
    outputs = model(**tokens)

    # Return the model outputs
    return outputs
    """

####################################
init_model()
input_text = "A Large Tree"
tokenize_text(input_text)
tokenize_text("A large tree")


tokenize_text("A")
tokenize_text("Large")
tokenize_text("Tree")

