

#backend = "tensorflow"
backend = "torch"

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
    import numpy as np

    # types
    Tensor = torch.Tensor
    SparseTensor = torch.sparse.Tensor # TODO: make a class for SparseTensor ?
    float32 = torch.float32
    
    sparse_coo_tensor = torch.sparse_coo_tensor

    # functions
    matmul = torch.matmul
    reshape = torch.reshape

    def multiply(x,y):
        return x*y

    def repeat(input, repeats, axis=None):
        return torch.repeat_interleave(input, repeats, dim=axis)
    
    def cast(x, dtype):
        if isinstance(x, np.ndarray):
            return cast(torch.from_numpy(x), dtype)
        if isinstance(x, list):
            try:
                x = np.array(x)
                x = torch.from_numpy(x)
            except TypeError:
                return [cast(item, dtype) for item in x]
            
            return cast(x, dtype)
        elif isinstance(x, dict):
            return {key:cast(x[key], dtype) for key in x}
        else:
            return x.to(dtype=dtype)

    def expand_dims(x, axis):
        return torch.unsqueeze(x, dim=axis)

    def clip_by_value(x, clip_value_min, clip_value_max):
        return torch.clamp(x, min=clip_value_min, max=clip_value_max)

    def concat(x, axis):
        return torch.cat(x, dim=axis)