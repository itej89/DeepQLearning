
from collections import deque


class QLearningParameters():

        #STANDARD Q-Learning EQUATION
        #----------------------------------------------------------------------------------------------
        #   Q(s, a) = ( 1 - learingRate ) * Q(s, a) + learingRate * ( reward + gamma * max(Q(s', a)) )
        #   (We have to learn this equation through a Neural Network)
        #----------------------------------------------------------------------------------------------

        #------------------------------------------------------------------
        #QLearning hyper Parameters
        #------------------------------------------------------------------
        #QLearning Parameters
        #Discount rate for the QLearning
        GAMMA=0.95

        #Explorarion vs exploitation
        #The Probability to chosing a random action over the current learned policy
        EPSILON = 1
        EPSILON_DECAY_VALUE = 0.95
        EPSILON_MIN_VALUE=0.1
        #------------------------------------------------------------------


        #------------------------------------------------------------------
        #NN hyper Parameters
        #------------------------------------------------------------------
        #Learning rate parameter for QLearning
        LEARNING_RATE=0.01

        #The batch size to train the neural network in each epoch 
        BATCH_SIZE=32
        #------------------------------------------------------------------


        #------------------------------------------------------------------
        #Update loop parameters
        #------------------------------------------------------------------
        #Total Number of training episodes for the agent to explore
        TOTAL_EPISODES=100

        #Maximum number of steps the agent can takes before exiting the environment
        MAX_STEPS_PER_EPISODE=200

        #Save Models for every so many episodes
        SAVE_MODEL_FOR_STEPS = 20

        #Render the Simulator every so many episodes
        RENDER_EVERY = 50

        #Model Save Path
        MODEL_PATH_SAVE = "./Models/Cartpole-V0"
        #------------------------------------------------------------------



        #Buffer to store all the states genrated by the agent 
        # by navigating the environment through different actions
        replayBuffer=deque(maxlen=20000)
