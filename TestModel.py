import numpy as np
from keras import models

from Environment import *
from QLearningParameters import *

MODEL_PATH = './Models/Cartpole-V0_74.h5'

if __name__ =='__main__':
    #Initialize an environment
    env = Environment()

    #Load the pretrained model
    model=models.load_model(MODEL_PATH)

    #Loop for 10 episodes
    for i_episode in range(10):

        #Get inital state
        state = env.reset().reshape(1, env.get_state_dimentions()[0])

        #Simulate for the max number of steps in an episode
        for t in range(QLearningParameters.MAX_STEPS_PER_EPISODE):
            env.render()

            #Take the best action prediction from the model for the current state
            best_action = np.argmax(model.predict(state)[0])

            #Control the environemnt with the best action form the policy
            next_state, reward, done, info = env.step(best_action)

            #Updat the current state
            next_state = next_state.reshape(1, env.get_state_dimentions()[0])
            state=next_state