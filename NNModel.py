import tensorflow as tf

from keras import models
from keras import layers
from keras.optimizers import Adam

from QLearningParameters import QLearningParameters
class DeepQModel:
 
    #Function to create the actual model
    # input_shape: this is the same size as the state of the environemnt
    def SpawnModel(self, input_shape, output_shape):
    
        model = models.Sequential()
        model.add(layers.Dense(48, activation='tanh', input_shape=input_shape))
        model.add(layers.Dense(output_shape,activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=QLearningParameters.LEARNING_RATE))
        return model