
from tensorflow.keras.layers import Input, Layer,Dropout
from tensorflow.keras import Model
from multiHead import MultiHeadAttention
from feedForward import FeedForward
from addNormalization import AddNormalization
from positionalEncoding import PositionEmbeddingFixedWeights

    
class EncoderLayer(Layer):

    def __init__(self, sequence_length,  h, d_k,d_v, d_model,d_ff, rate, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.multihead_attention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff=d_ff,d_model=d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def call(self, x, padding_mask, training):
        #Multi-Head attention layer
        multihead_output = self.multihead_attention(x,x,x, padding_mask)
        #Expected output shape = (batch_size, sequence_length, d_model)

        #Add in a dropout layer
        multihead_output = self.dropout1(multihead_output, training= training)

        #Followed by an add & Norm layer
        addnorm_output =  self.add_norm1(x,multihead_output)
        #Expected output shape = (batch_size, sequence_length, d_model)

        #Fully connected Layer
        feed_forward_output = self.feed_forward(addnorm_output)

        #Add in another dropout layer
        feed_forward_output = self.dropout2(feed_forward_output, training= training)

        #Followed by another Add&Norm
        return self.add_norm2(addnorm_output, feed_forward_output)

class Encoder(Layer):
    def __init__(self, vocab_size, sequence_length, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super().__init__(**kwargs)
        self.pos_encoding = PositionEmbeddingFixedWeights(sequence_length, vocab_size, d_model)

        self.dropout = Dropout(rate)
        self.encoder_layer = [EncoderLayer(sequence_length,h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, input_sentence, padding_mask, training):

            #positional encoding

            pos_encoding_output = self.pos_encoding(input_sentence)
            # Expected output shape = (batch_size, sequence_length, d_model)

            #Add in a dropout layeer

            x  = self.dropout(pos_encoding_output, training=training)

            for i,layer in  enumerate(self.encoder_layer):
                x = layer(x, padding_mask, training)
            return x