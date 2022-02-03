import math

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras.layers import Concatenate, Dropout, GRU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

tf.compat.v1.disable_eager_execution()
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.layers import LSTM, BatchNormalization, Activation, Bidirectional

METRICS = [
    metrics.BinaryAccuracy(name='accuracy'),
    metrics.Precision(name='precision'),
    metrics.Recall(name='recall'),
    metrics.AUC(name='auc'),
]


def make_model(demo_feature_count, temp_feature_count, temp_count, tr_flag):
    input_demo = Input(shape=(demo_feature_count))
    input_temp = Input(shape=(temp_count, temp_feature_count))

    lstm = LSTM(units=8, return_sequences=True, trainable= tr_flag)(input_temp)
    lstm = LSTM(units=16, activation='tanh', trainable= tr_flag)(lstm)

    dense = Dense(8, trainable= tr_flag)(input_demo)
    dense = Activation('relu', trainable= tr_flag)(dense)
    dense = Dense(16, trainable= tr_flag)(dense)
    dense = Activation('relu', trainable= tr_flag)(dense)

    concat = Concatenate()([lstm, dense])

    attrition = Dense(32)(concat)
    attrition = Activation('relu')(attrition)
    attrition = Dense(16)(attrition)
    attrition = Activation('relu')(attrition)
    output_attrition = Dense(1, activation='sigmoid', name="attrition")(attrition)

    outcome = Dense(32)(concat)
    outcome = Activation('relu')(outcome)
    outcome = Dense(16)(outcome)
    outcome = Activation('relu')(outcome)
    output_outcome = Dense(1, activation='sigmoid', name="outcome")(outcome)

    # model = Model(inputs=[input_demo, input_temp], outputs=[output_attrition, output_outcome])
    model = Model(inputs=[input_demo, input_temp], outputs=[output_outcome])

    #model.compile(optimizer="adam", loss=['binary_crossentropy', 'binary_crossentropy'], metrics=METRICS)
    model.compile(optimizer="adam", loss=['binary_crossentropy'], metrics=METRICS)

    # summarize layers
    # print(model.summary())
    # plot graph
    #plot_model(model, to_file='recurrent_neural_network.png')

    return model
