from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPool1D, Dropout, Flatten, Dense, LSTM, Bidirectional, Add, LayerNormalization

from tfomics.layers import MultiHeadAttention


def CNN(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=4, dense_units=512, num_out=12):

    inputs = Input(shape=in_shape)
    nn = Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = BatchNormalization()(nn)
    nn = Activation(activation, name='conv_activation')(nn)
    nn = MaxPool1D(pool_size=pool_size)(nn)
    nn = Dropout(0.1)(nn)

    nn = Flatten()(nn)

    nn = Dense(dense_units, use_bias=False)(nn)
    nn = BatchNormalization()(nn)
    nn = Activation('relu')(nn)
    nn = Dropout(0.5)(nn)

    outputs = Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)


def CNN_LSTM(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=4, lstm_units=128, dense_units=512, num_out=12):
    
    inputs = Input(shape=in_shape)
    nn = Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = BatchNormalization()(nn)
    nn = Activation(activation, name='conv_activation')(nn)
    nn = MaxPool1D(pool_size=pool_size)(nn)
    nn = Dropout(0.1)(nn)

    forward = LSTM(lstm_units//2, return_sequences=True)
    backward = LSTM(lstm_units//2, activation='relu', return_sequences=True, go_backwards=True)
    nn = Bidirectional(forward, backward_layer=backward)(nn)
    nn = Dropout(0.1)(nn)

    nn = Flatten()(nn)

    nn = Dense(dense_units, use_bias=False)(nn)
    nn = BatchNormalization()(nn)
    nn = Activation('relu')(nn)
    nn = Dropout(0.5)(nn)

    outputs = Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)


def CNN_ATT(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=4, heads=8, key_size=64, dense_units=512, num_out=12):

    inputs = Input(shape=in_shape)
    nn = Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = BatchNormalization()(nn)
    nn = Activation(activation, name='conv_activation')(nn)
    nn = MaxPool1D(pool_size=pool_size)(nn)
    nn = Dropout(0.1)(nn)

    nn, w = MultiHeadAttention(num_heads=heads, d_model=key_size)(nn, nn, nn)
    nn = Dropout(0.1)(nn)

    nn = Flatten()(nn)

    nn = Dense(dense_units, use_bias=False)(nn)
    nn = BatchNormalization()(nn)
    nn = Activation('relu')(nn)
    nn = Dropout(0.5)(nn)

    outputs = Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)


def CNN_LSTM_ATT(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=4, lstm_units=128, heads=8, key_size=64, dense_units=512, num_out=12):

    inputs = Input(shape=in_shape)
    nn = Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = BatchNormalization()(nn)
    nn = Activation(activation, name='conv_activation')(nn)
    nn = MaxPool1D(pool_size=pool_size)(nn)
    nn = Dropout(0.1)(nn)

    forward = LSTM(lstm_units//2, return_sequences=True)
    backward = LSTM(lstm_units//2, activation='relu', return_sequences=True, go_backwards=True)
    nn = Bidirectional(forward, backward_layer=backward)(nn)
    nn = Dropout(0.1)(nn)
    
    nn, w = MultiHeadAttention(num_heads=heads, d_model=key_size)(nn, nn, nn)
    nn = Dropout(0.1)(nn)

    nn = Flatten()(nn)

    nn = Dense(dense_units, use_bias=False)(nn)
    nn = BatchNormalization()(nn)
    nn = Activation('relu')(nn)
    nn = Dropout(0.5)(nn)

    outputs = Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)


def CNN_TRANS(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=4, num_layers=1, heads=8, key_size=64, dense_units=512, num_out=12):
    
    inputs = Input(shape=in_shape)
    nn = Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = BatchNormalization()(nn)
    nn = Activation(activation, name='conv_activation')(nn)
    nn = MaxPool1D(pool_size=pool_size)(nn)
    nn = Dropout(0.1)(nn)
    
    # sizes don't align; num_filters =/= key_size
    
    nn = LayerNormalization(epsilon=1e-6)(nn)
    for i in range(num_layers):
        nn2,_ = MultiHeadAttention(d_model=key_size, num_heads=heads)(nn, nn, nn)
        nn2 = Dropout(0.1)(nn2)
        nn = Add()([nn, nn2])
        nn = LayerNormalization(epsilon=1e-6)(nn)
        nn2 = Dense(32, activation='relu')(nn)
        nn2 = Dropout(0.2)(nn2)
        nn2 = Dense(key_size)(nn2)
        nn2 = Dropout(0.1)(nn2)
        nn = Add()([nn, nn2])
        nn = LayerNormalization(epsilon=1e-6)(nn)
    
    nn = Flatten()(nn)

    nn = Dense(dense_units, use_bias=False)(nn)
    nn = BatchNormalization()(nn)
    nn = Activation('relu')(nn)
    nn = Dropout(0.5)(nn)

    outputs = Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)


def CNN_LSTM_TRANS(in_shape=(200, 4), num_filters=32, batch_norm=True, activation='relu', pool_size=4, num_layers=1, heads=8, key_size=64, dense_units=512, num_out=12):
    
    inputs = Input(shape=in_shape)
    nn = Conv1D(filters=num_filters, kernel_size=19, use_bias=False, padding='same')(inputs)
    if batch_norm:
        nn = BatchNormalization()(nn)
    nn = Activation(activation, name='conv_activation')(nn)
    nn = MaxPool1D(pool_size=pool_size)(nn)
    nn = Dropout(0.1)(nn)
    
    forward = LSTM(key_size // 2, return_sequences=True)
    backward = LSTM(key_size // 2, activation='relu', return_sequences=True, go_backwards=True)
    nn = Bidirectional(forward, backward_layer=backward)(nn)
    nn = Dropout(0.1)(nn)
    
    nn = LayerNormalization(epsilon=1e-6)(nn)
    for i in range(num_layers):
        nn2,_ = MultiHeadAttention(d_model=key_size, num_heads=heads)(nn, nn, nn)
        nn2 = Dropout(0.1)(nn2)
        nn = Add()([nn, nn2])
        nn = LayerNormalization(epsilon=1e-6)(nn)
        nn2 = Dense(32, activation='relu')(nn)
        nn2 = Dropout(0.2)(nn2)
        nn2 = Dense(key_size)(nn2)
        nn2 = Dropout(0.1)(nn2)
        nn = Add()([nn, nn2])
        nn = LayerNormalization(epsilon=1e-6)(nn)
    
    nn = Flatten()(nn)

    nn = Dense(dense_units, use_bias=False)(nn)
    nn = BatchNormalization()(nn)
    nn = Activation('relu')(nn)
    nn = Dropout(0.5)(nn)

    outputs = Dense(num_out, activation='sigmoid')(nn)

    return Model(inputs=inputs, outputs=outputs)
















