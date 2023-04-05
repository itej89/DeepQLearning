from QLearningParameters import QLearningParameters
from Environment import Environment
from NNModel import DeepQModel
from QLearning import RLTrainer

import matplotlib.pylab as plt

if __name__=="__main__":

    #Model to learn the Q-function
    DeepQModel = DeepQModel()

    #Environment to be controlled
    _Environment = Environment()

    #Initialize Q-learning trainer with untrained Q-learning nn model and the environment
    Trainer=RLTrainer(DeepQModel, _Environment)

    #Run Reinforcement training
    for episode in range(QLearningParameters.TOTAL_EPISODES):
        
        #get environment state dimention
        state_dim = _Environment.get_state_dimentions()[0]

        #Reset environment on each episode
        initialState=_Environment.reset().reshape(1, _Environment.get_state_dimentions()[0])

        #Render the environment every few episodes
        #For performance reasons
        render = False
        if episode % QLearningParameters.RENDER_EVERY == 0:
            render = True

        #Explore the environment and train the model for an episode
        Trainer.explore_environment(initialState, episode, render)



    #Plot rewards recieved at each episode during training 
    episodes = list(Trainer.RewardList.keys())
    rewards  = list(Trainer.RewardList.values())
    plt.plot(episodes,rewards)
    plt.show()