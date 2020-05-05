import numpy as np
from recogym.agents import Agent

num_products = 10

num_offline_users = 5
num_online_users = 5

debug = False

class StaticAgent(Agent):

    def __init__(self, config):

        super(StaticAgent, self).__init__(config)  # Set number of products as an attribute of the Agent.

        self.organic_views = np.zeros(
            self.config.num_products)  # Track number of times each item viewed in Organic session.

        self.act_counter = 0
        self.train_counter = 0

    def train(self, observation, action, reward, done):

        # Train method learns from a tuple of data. This method can be called for offline or online learning
        # Adding organic session to organic view counts.

        if observation:

            for session in observation.sessions():  # -- LOOP

                if debug:
                    print("\n-------------- TRAIN START --------------")
                    print(f"train () {self.train_counter} :::: reward {reward}")

                self.organic_views[session['v']] += 1

                if debug:
                    print(f"train () {self.train_counter} :::: self.organic_views {self.organic_views}")

                self.train_counter += 1

                if debug:
                    print("-------------- TRAIN END --------------\n")

    def act(self, observation, reward, done):

        if debug:
            print("\n-------------- ACT START --------------")

        # An act method takes in an observation, which could either be `None` or an Organic_Session
        # and returns a integer between 0 and num_products indicating which product the agent recommends.

        if debug:
            print(f"act () {self.act_counter} :::: get reward {reward}")
            print(f"act () {self.act_counter} :::: get observation sessions {observation.sessions()}")
            print(f"act () {self.act_counter} :::: have organic_views {self.organic_views}")
            print(f"act () {self.act_counter} :::: have sum(self.organic_views) {sum(self.organic_views)}")

        prob = self.organic_views / sum(self.organic_views)

        if debug:
            print(f"act () {self.act_counter} :::: calc prob {prob}")
            print(f"act () {self.act_counter} :::: have num_products {num_products}")

        # Choosing action randomly in proportion with number of views.

        action = 1

        if debug:
            print(f"act () {self.act_counter} :::: return action {action}")
            print(f"act () {self.act_counter} :::: return prob[action] {prob[action]}")

        self.act_counter += 1

        if debug:
            print("-------------- ACT END --------------\n")

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': prob[action]
            }
        }

# %%
