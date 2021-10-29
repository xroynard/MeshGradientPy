
backend = "tensorflow"
#backend = "torch"

if backend == "tensorflow":
    import tensorflow as tf
    import tf.Tensor as Tensor
    import tf.sparse.SparseTensor as SparseTensor
    import tf.sparse.sparse_dense_matmul as MatMul

elif backend == "torch":
    import torch
    ### TODO: the same for PyTorch