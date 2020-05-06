import numpy as np

from recogym import Configuration
from recogym.agents import Agent

single_action_args = {
    'num_products': 10,
    'random_seed': np.random.randint(2 ** 31 - 1),
    'with_ps_all': False,
}

class SingleActionAgent(Agent):

    def __init__(self, config=Configuration(single_action_args)):

        print(f"SingleActionAgent %%%% num_products: {config.num_products}")

        super(SingleActionAgent, self).__init__(config)  # Set number of products as an attribute of the Agent.

        self.organic_views = np.zeros(
            self.config.num_products)  # Track number of times each item viewed in Organic session.

        self.act_counter = 0
        self.train_counter = 0

    def train(self, observation, action, reward, done):

        if observation:

            for session in observation.sessions():  # -- LOOP

                self.organic_views[session['v']] += 1

    def act(self, observation, reward, done):

        prob = self.organic_views / sum(self.organic_views)

        action = 1  # always return the same product id

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': prob[action]
            }
        }

# %%
