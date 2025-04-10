import tensorflow as tf 
from tensorflow import convert_to_tensor, string, math, cast, matmul, float32
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
from tensorflow.data import Dataset
from keras.activations import softmax
import numpy as np
from numpy import random
import matplotlib.pyplot as plt



class DotProductAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, queries, keys, values, d_k, mask=None):
        scores = matmul(queries, keys, transpose_b=True) /  math.sqrt(cast(d_k, float32))

        if mask is not None:
            scores += -1e9 * mask
        
        weights = softmax(scores)

        return matmul(weights, values)


d_k = 64 # Dimensionality of the linearly projected queries and keys
d_v = 64 # Dimensionality of the linearly projected values
batch_size = 64 # Batch size from the training process

input_seq_length = 5 # Maximum length of the input sequence
 
queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))


attention = DotProductAttention()
print(attention(queries, keys, values, d_k=d_k))

