

# How to get a hash of a tensor (thanks chatgpt)

numpy_array = tensor.numpy()
checksum = hash(numpy_array.tobytes())
hex_checksum = format(checksum, 'x')

