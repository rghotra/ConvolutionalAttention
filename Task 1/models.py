import tensorflow as tf
from keras import layers, Model

class CNN_ATT(tf.keras.Model):

    def __init__(self, num_filters=32, batch_norm=True, activation='relu', pool_size=4, heads=16, key_size=64, dense_units=512):
        super(CNN_ATT, self).__init__()
        
        # Convolution
        self.conv_layer = layers.Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')
        self.batch_norm = layers.BatchNormalization() if batch_norm else None
        self.activation = layers.Activation(activation, name='conv_activation')
        self.max_pool = layers.MaxPool1D(pool_size=pool_size)

        # Multi-Head Attention
        self.attention = layers.MultiHeadAttention(num_heads=heads, key_dim=key_size)

        # Feed Forward
        self.dense = layers.Dense(dense_units, use_bias=False)

    def call(self, inputs):
        
        nn = self.conv_layer(inputs)
        if batch_norm is not None:
            nn = self.batch_norm(nn)
        nn = self.activation(nn)
        nn = self.max_pool(nn)
        nn = layers.Dropout(0.1)(nn)
        
        nn, w = self.attention(nn, nn, return_attention_scores=True)
        nn = layers.Dropout(0.1)(nn)
        
        nn = layers.Flatten()(nn)
        
        nn = self.dense(nn)
        nn = layers.BatchNormalization()(nn)
        nn = layers.Activation('relu')(nn)
        nn = layers.Dropout(0.5)(nn)
        
        nn = layers.Dense(12, activation='sigmoid')
        
        return nn
