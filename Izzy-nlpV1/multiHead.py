
from tensorflow.keras.layers import Dense, Layer
from dotProductAttention import DotProductAttention
from numpy import random, array
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32

class MultiHeadAttention(Layer):
    def __init__(self, h, d_k, d_v, d_model, **kwargs):
        super().__init__(**kwargs)
        self.attention =  DotProductAttention() #Scaled DotProductAttention
        self.heads =  h 
        self.d_k = d_k
        self.d_v = d_v
        self.w_q = d_k
        self.w_k = d_k
        self.w_v = d_v
        self.w_o = d_model


    #We are now going to reshape the linearly projected queries, keys, and valuess in such a manner as to allow the attention heads to be computed in parallel
        
    def reshape_tensor(self, x, heads, flag):
        if flag:
            #Tensor shape after reshaping and transposing:
            #(batch_size, heads, seq_length, -1)

            x =  reshape(x, shape=(shape(x)[0], shape(x)[1],heads, -1) )
            x = transpose(x, perm=(0,2,1,3)) 

        else:
            #Reverting the reshaping and transposing operations:
            #(batch_size, seq_length, d_modl)
            x = transpose(x, perm=(0,2,1,3)) 
            x = reshape(x, shape=(shape(x)[0], shape(x)[1],-1) )
        return x 
    
    def build(self, input_shape):
        # input_shape = (batch_size, sequence_length, d_model)
        self.w_q = Dense(self.heads * self.d_k)
        self.w_k = Dense(self.heads * self.d_k)
        self.w_v = Dense(self.heads * self.d_v)
        self.w_o = Dense(self.w_o)
        super().build(input_shape)

    def call(self, queries, keys, values, mask=None):
        # Rearrange the queries to be able to compute all heads in parallel
        q_reshaped = self.reshape_tensor(self.w_q(queries),self.heads, True)

        k_reshaped = self.reshape_tensor(self.w_k(keys),self.heads, True)

        v_reshaped = self.reshape_tensor(self.w_v(values),self.heads, True)
        
        # Compute the multi-head attention output using the reshaped queries, keys,
# and values
        o_reshaped = self.attention(q_reshaped,k_reshaped,v_reshaped, d_k=self.d_k, mask=mask)
        # Resulting tensor shape: (batch_size, heads, input_seq_length, -1)

         # Rearrange back the output into concatenated form
        output = self.reshape_tensor(o_reshaped, self.heads, False) # Resulting tensor shape: (batch_size, input_seq_length, d_v)
# Apply one final linear projection to the output to generate the multi-head # attention. Resulting tensor shape: (batch_size, input_seq_length, d_model) 
        return self.w_o(output)
    

h = 8 # Number of self-attention heads
d_k = 64 # Dimensionality of the linearly projected queries and keys 
d_v = 64 # Dimensionality of the linearly projected values
d_model = 512 # Dimensionality of the model sub-layers' outputs 
batch_size = 64 # Batch size from the training process

input_seq_length = 5 # Maximum length of the input sequence
queries = random.random((batch_size, input_seq_length, d_k))
keys = random.random((batch_size, input_seq_length, d_k))
values = random.random((batch_size, input_seq_length, d_v))

multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)

print(multihead_attention(queries, keys, values))