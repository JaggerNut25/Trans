from tensorflow.keras.layers import LayerNormalization, Layer

class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = LayerNormalization() #LayerNormalization

    def call(self, x, sublayer_x):
        #The sublayeer input and output need to be the same shape to be summed
        add = x + sublayer_x

        #Apply layer normalization to thee sum
        return self.layer_norm(add)
    