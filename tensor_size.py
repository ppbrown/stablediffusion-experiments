

# Snippet to calculate memory size used by a tensor.
# If you just call the builtin size thing, it only gives size of the
# top level struct or whatever.

def get_tensor_memory_size(tensor):
    element_size = tensor.element_size()
    numel = tensor.numel()
    total_memory = element_size * numel
    return total_memory

