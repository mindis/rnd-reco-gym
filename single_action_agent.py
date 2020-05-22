import numpy as np

from recogym import Configuration
from recogym.agents import Agent

single_action_args = {
    'num_products': 1000,
    'random_seed': np.random.randint(2 ** 31 - 1),
    'with_ps_all': False,
}

debug = False

class SingleActionAgent(Agent):

    def __init__(self, config=Configuration(single_action_args)):
        super(SingleActionAgent, self).__init__(config)
        print(f"SingleActionAgent %%%% num_products: {config.num_products}")

        self.organic_views = np.zeros(self.config.num_products)

        self.act_counter = 0
        self.train_counter = 0

    def train(self, observation, action, reward, done):

        self.train_counter += 1

        if debug:
            print(f"\nSingleActionAgent %%%% TRAIN {self.train_counter} observation: {observation.sessions()}")


    def act(self, observation, reward, done):

        self.act_counter += 1

        if debug:
            print(f"\nSingleActionAgent %%%% ACT {self.act_counter} act observation: {observation.sessions()}")

        prob = self.organic_views / sum(self.organic_views)

        action = 1  # always return the same product id

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': prob[action]
            }
        }

