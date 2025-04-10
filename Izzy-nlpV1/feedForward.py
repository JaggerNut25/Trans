
from tensorflow.keras.layers import Layer, Dense, ReLU


class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super().__init__(**kwargs)
        self.fully_connected_1 = Dense(d_ff) #First fully connected layeer
        self.fully_connected_2 = Dense(d_model) #Second fully connected
        self.activation = ReLU() #ReLU activation Layer 

    def call(self, x):
        #The input is passed first through the first Linear transformation then trhrough thee second with the ReLU activation function
        x_fc1 = self.fully_connected_1(x)

        return self.fully_connected_2(self.activation(x_fc1))