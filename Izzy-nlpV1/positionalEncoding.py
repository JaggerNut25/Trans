import tensorflow as tf 
from tensorflow import convert_to_tensor, string
from tensorflow.keras.layers import TextVectorization, Embedding, Layer
from tensorflow.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

#Lets define the maximum length we want to fixed for the vectors of sequence
output_seq_len = 5

#TextVec of Keras require like a maxim vocab size
vocab_size = 10

sentences = [["I am a robot"], ["you too robot"]]

sentences_data = Dataset.from_tensor_slices(sentences)

#creation of the Textvectorization layer

vectorize_layer = TextVectorization(output_sequence_length=output_seq_len, max_tokens=vocab_size)

#Train the layer to create a dictionary
vectorize_layer.adapt(sentences_data)

#convert all sentences to tensor
word_tensors = convert_to_tensor(sentences, dtype=tf.string)

#use the word tensors to get vectorized phrases

vectorized_words = vectorize_layer(word_tensors)

print("vocbulary: ", vectorize_layer.get_vocabulary())
print("vectorized words: ", vectorized_words)

#Lets continue with the Embedding layer, but before doing this wr need to specify the maximum value of an integer to map cuz the layer maps the preevious integers to random numbers

output_length = 6
word_embedding_layer = Embedding(vocab_size, output_length)
embedded_words  = word_embedding_layer(vectorized_words)


print(embedded_words)

#Positional Embeddings

position_embedding_layer = Embedding(output_seq_len, output_length)
position_indices = tf.range(output_seq_len)
embedded_indices = position_embedding_layer(position_indices)

#In transformers the final output of Positional encoding layer is the sum of word embeddings and the position embeddings

final_output_embedding = embedded_words + embedded_indices

print("Final output: ", final_output_embedding)


#Lets Subclassing the Keras Embedding Layer

class PositionEmbeddingLayer(Layer):
    def __init__(self, seq_len, vocab_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.word_embedding_layer = Embedding(input_dim= vocab_size, output_dim= output_dim)
        self.position_embedding_layer = Embedding(input_dim=seq_len,output_dim=output_dim)

    
    def call(self, inputs):
        position_indices= tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices= self.position_embedding_layer(position_indices)
    
        return embedded_words + embedded_indices
    
#Use this one(Omo this code is just for educational purpose ooo, do your own if anything happen)
class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, seq_len, vocab_size, output_dim, **kwargs):
        super().__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size,output_dim)
        pos_embedding_matrix = self.get_position_encoding(seq_len, output_dim)
        self.word_embedding_layer = Embedding(input_dim= vocab_size, output_dim= output_dim, weights=[word_embedding_matrix])
        self.position_embedding_layer = Embedding(input_dim=seq_len,output_dim=output_dim, weights=[pos_embedding_matrix])

    
    def get_position_encoding(self, seq_len,d, n=10000):
            p= np.zeros((seq_len, d))
            for k in range(seq_len):
                 for i in np.arange(int(d/2)):
                    denominator = np.power(n, 2*i/d)
                    p[k, 2*i]= np.sin(k / denominator)
                    p[k, 2*i + 1]= np.cos(k / denominator)
            return p

        
    
    
    
    
    def call(self, inputs):
        position_indices= tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices= self.position_embedding_layer(position_indices)
    
        return embedded_words + embedded_indices
    
my_embedding_layer = PositionEmbeddingLayer(output_seq_len,vocab_size,output_length)

embedded_layer_output = my_embedding_layer(vectorized_words)

print("Output from my_embedded_layer: ", embedded_layer_output)

attnisallyouneed_embedding = PositionEmbeddingFixedWeights(output_seq_len, vocab_size, output_length)
attnisallyouneed_output = attnisallyouneed_embedding(vectorized_words)

print("Output from my_embedded_layer: ", attnisallyouneed_output)

