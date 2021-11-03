

backend = "tensorflow"
#backend = "torch"

if backend == "tensorflow":
    import tensorflow as tf

    # types
    Tensor = tf.Tensor
    SparseTensor = tf.sparse.SparseTensor
    float32 = tf.float32

    # functions
    matmul = tf.sparse.sparse_dense_matmul
    reshape = tf.reshape

    multiply = tf.math.multiply
    repeat = tf.repeat
    cast = tf.cast
    expand_dims = tf.expand_dims
    clip_by_value = tf.clip_by_value
    concat = tf.concat

elif backend == "torch":
    import torch

    # types
    Tensor = torch.Tensor
    SparseTensor = torch.Tensor # TODO: make a class for SparseTensor ?
    float32 = torch.float32

    # functions
    matmul = torch.matmul
    reshape = torch.reshape

    def multiply(x,y):
        return x*y

    def repeat(input, repeats, axis=None):
        return torch.repeat_interleave(input, repeats, dim=axis)
    
    def cast(x, dtype):
        return x.to(dtype=dtype)

    def expand_dims(x, axis):
        return torch.unsqueeze(x, dim=axis)

    def clip_by_value(x, clip_value_min, clip_value_max):
        return torch.clamp(x, min=clip_value_min, max=clip_value_max)

    def concat(x, axis):
        return torch.cat(x, dim=axis)