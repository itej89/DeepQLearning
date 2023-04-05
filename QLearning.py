

import random
import numpy as np

from QLearningParameters import QLearningParameters

from tqdm import tqdm


class RLTrainer:
    def __init__(self, nnmodel, environment):

        self.RewardList = {}
        self.env=environment
        
        #The neural network is used to generate the function Q(s,a)
        # Input to the NN is the state from the environment
        # Output of the NN is the set of Q(s,ai) values for all actions i=1..n

        #Create the training Network
        #This is the final network that will be learning the Q-function for the environment
        self.PolicyNetwork=nnmodel.SpawnModel(self.env.get_state_dimentions(), self.env.get_descrete_action_count())

        #Create the target network
        #Since we will be updating thte "training network" weights very frequently
        # we can not use the same network as a stable policy for learning
        # So we create a clone network for the training network whose weights will be 
        # updated less frequently by copying the training network
        #We will be using this network as a stable temporary policy for taking actions
        # to move to new states
        self.PolicyNetwork_Stable_Train=nnmodel.SpawnModel(self.env.get_state_dimentions(), self.env.get_descrete_action_count())

        #Make sure we clone the target network with the train network's weights
        self.PolicyNetwork_Stable_Train.set_weights(self.PolicyNetwork.get_weights())

   
    #Method that choses the next action
    # the method returns a random action with epsilon probability
    # other times it exploits the learned policy learned and gives the next action to take
    def env_explore_exploit(self,state):
        
        #Chose the minimum of epsilon for the entire training sequence
        QLearningParameters.EPSILON = max(QLearningParameters.EPSILON_MIN_VALUE, QLearningParameters.EPSILON)

        if np.random.rand(1) < QLearningParameters.EPSILON:
            #Sample random actin from list of possible actions
            action = np.random.randint(0, self.env.get_descrete_action_count())
        else:
            #Use learned policy to get the  action scores and
            # chose the actin with the maximum reward
            action=np.argmax(self.PolicyNetwork.predict(state)[0])
        return action

    def get_samples(self):
          #Check if the replay has enough samples to do 
        # the batch training on the neural network
        if len(QLearningParameters.replayBuffer) < QLearningParameters.BATCH_SIZE:
            return None

        #If sufficient number of samples are availabl on the replay buffer
        # sample states from the replay buffer 
        return random.sample(QLearningParameters.replayBuffer,QLearningParameters.BATCH_SIZE)

    #Extract states and next state pairs from samples
    # States and next states are used to get Q(s,a) and Q(s',a) for the Q-Learnng
    def get_train_data_batch(self, moves_batch):

        #States
        state_pairs = []

        #Loop through the state information sampled from the environment
        for move in moves_batch:
            state, action, reward, new_state, done = move
            state_pairs.append([state,new_state])

        state_pairs = np.array(state_pairs, dtype=np.float32)
        #Create an array of sampled states
        states = state_pairs[:,0].reshape(QLearningParameters.BATCH_SIZE, 4)

        #Create an array of the next states for the sampled states
        nextSates = state_pairs[:,1].reshape(QLearningParameters.BATCH_SIZE, 4)

        return (states, nextSates)

    def update_qnn_function(self):
        
        batch_moves = self.get_samples()

        if batch_moves == None:
            return
        
        states, nextSates = self.get_train_data_batch(batch_moves)

        #Get Q(s,a) of all actions for the input states
        #This gives Q(s,a) present
        Q_values = self.PolicyNetwork.predict(states)

        #Get Q(s,a) of all actions for the input states
        #This gives max(Q(s',a)) for the future state
        Next_State_Q_Values=self.PolicyNetwork_Stable_Train.predict(nextSates)

        #--------------------------------------------------------
        #Q-learning update code
        #--------------------------------------------------------
        for i in range(QLearningParameters.BATCH_SIZE):
            #This is the actual values from the environment
            state, action, reward, new_state, done = batch_moves[i]

            #get action prediction of the each sample
            Q_Value = Q_values[i]

            #if done then set the reward of the respective state as its Q value
            if done:
                Q_Value[action] = reward
            #if not done, then perform the Q update to the sample (state, action) value
            else:
                #get the (next_state,action) pair with maximum score
                Q_Max_Next_State = max(Next_State_Q_Values[i])

                #Perform the Q update.
                # Observe that the remaining part of 
                # the "Q update-equation" will be performed by the NN.
                Q_Value[action] = (reward + QLearningParameters.GAMMA * Q_Max_Next_State)
        
        # fit the network with updates Q values
        self.PolicyNetwork.fit(states, Q_values, epochs=1, verbose=0)
        #--------------------------------------------------------


    def explore_environment(self, state, episode, render):

        TotalReward = 0

        for i in tqdm(range(QLearningParameters.MAX_STEPS_PER_EPISODE+1)):
            action = self.env_explore_exploit(state)

            #show the animation every 50 episodes
            if render:
                self.env.render()
            
            #Take action from the current state
            next_state, reward, done, _ = self.env.step(action)

            next_state = next_state.reshape(1, self.env.get_state_dimentions()[0])

            cart_position = next_state[0][0]
            cart_velocity = next_state[0][1]
            pole_angle    = next_state[0][2]
            pole_velocity = next_state[0][3]

            if .2095 <= pole_angle >= -.2095 and 4.8 <= cart_position >= -4.8 :
                reward += 10


            QLearningParameters.replayBuffer.append([state, action, reward, next_state, done])

            self.update_qnn_function()

            TotalReward += reward

            state = next_state

            if done:
                break

               
        if  done:
            self.PolicyNetwork.save(QLearningParameters.MODEL_PATH_SAVE+'_{}.h5'.format(episode))

        self.RewardList[episode] = TotalReward
        self.PolicyNetwork_Stable_Train.set_weights(self.PolicyNetwork.get_weights())

        QLearningParameters.EPSILON -= QLearningParameters.EPSILON_DECAY_VALUE



