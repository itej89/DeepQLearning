import gym

class Environment:

    OPNE_AI_MODEL_NAME = 'CartPole-v0'

    def __init__(self):
        self.env = gym.make(self.OPNE_AI_MODEL_NAME)
    
    def get_descrete_action_count(self):
        return self.env.action_space.n

    def get_state_dimentions(self):
        return self.env.observation_space.shape

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()

    def step(self, action):
       return self.env.step(action)