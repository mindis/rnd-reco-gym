import numpy as np
from numpy.random.mtrand import RandomState
from sklearn.linear_model import LogisticRegression

from recogym import DefaultContext, Observation
from recogym.agents import Agent
from recogym.envs.session import OrganicSessions
from recogym.agents import FeatureProvider

debug = False


def build_train_data(logs, feature_provider):

    user_states, actions, rewards, proba_actions = [], [], [], []

    current_user = None
    
    for _, row in logs.iterrows():
        
        if current_user != row['u']:
            # User has changed: start a new session and reset user state.
            current_user = row['u']
            sessions = OrganicSessions()
            feature_provider.reset()

        context = DefaultContext(row['u'], row['t'])

        if row['z'] == 'organic':
            sessions.next(context, row['v'])

        else:

            # For each bandit event, generate one observation for the user state,
            # the taken action the obtained reward and the used probabilities.
            
            feature_provider.observe(Observation(context, sessions))

            user_states.append(feature_provider.features(None).copy())

            actions.append(row['a'])
            rewards.append(row['c'])

            proba_actions.append(row['ps'])

            if debug:
                print(f"\nLikelihoodAgent build_train_data() "
                      f"\nactions {actions} "
                      f"\nrewards {rewards}")

            if debug:
                print(f"\nLikelihoodAgent build_train_data() row['a'] {row['a']} row['c'] {row['c']}")


        # Start a new organic session.
            
            sessions = OrganicSessions()

    return np.array(user_states), np.array(actions).astype(int), np.array(rewards), np.array(proba_actions)




class LikelihoodAgent(Agent):

    def __init__(self, feature_provider, seed=43):

        self.feature_provider = feature_provider
        self.random_state = RandomState(seed)
        self.model = None

        if debug:
            print(f"\nLikelihoodAgent INIT num_products {self.feature_provider.config.num_products}")

    @property
    def num_products(self):

        # if debug:
        #     print(f"\nLikelihoodAgent num_products {self.feature_provider.config.num_products}")

        return self.feature_provider.config.num_products




    def _create_features(self, user_state, action):

        # Look at the data and see how it maps into the features - which is the combination of the history
        # and the actions and the label, which is clicks.
        # Note that only the bandit events correspond to records in the training data.

        # To make a personalization, it is necessary to cross the action and history features.
        # _Why_ ?  We do the simplest possible to cross an element-wise Kronecker product.

        """Create the features that are used to estimate the expected reward from the user state"""

        # print(f"\nLikelihoodAgent train() features size {len(user_state) * self.num_products}")

        features = np.zeros(len(user_state) * self.num_products)

        # if debug:
        #     print(f"\nLikelihoodAgent Create the features that are used to "
        #           f"estimate the expected reward from the user state "
        #           f"\nsize {len(user_state) * self.num_products}"
        #           f"\nfeatures {features}")

        features[action * len(user_state): (action + 1) * len(user_state)] = user_state

        # if debug:
        #     print(f"\nLikelihoodAgent _create_features() action {action} "
        #           f"\nfuture index start {action * len(user_state)}"
        #           f"\nfuture index end {(action + 1) * len(user_state)}"
        #           f"\nuser_state {user_state}"
        #           f"\nRETURN features {features}")

        return features



        # The "features" are represented by matrix of organic product views where horizontal position is product id 0-9.
        # The every line is organic views and vertical offset is action id / recommendation product (7 for first matrix and 2 for the next one). Looks they call it "Kronecker product".
        #
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 1. 2. 1. 0. 0. 0. 1. 2. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 1. 2. 1. 0. 0. 0. 1. 2. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        #
        # .... there are 59 such matrices of observations in this configuration.
        #
        # The "rewards" is represented by array of size 59, 1 = click at bandit offer, there were 59 bandit actions 57 failures, 2 successful,  1 = click on bandit offer:
        #
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
        # 1. 0. 0. 0. 0. 0. 0. 0. 0.

    def train(self, logs):

        print(f"\nLikelihoodAgent train() num_products {self.feature_provider.config.num_products}")

        user_states, actions, rewards, proba_actions = build_train_data(logs, self.feature_provider)  # ---- build_train_data

        features = np.vstack([
            self._create_features(user_state, action)
            for user_state, action in zip(user_states, actions)
        ])

        self.model = LogisticRegression(solver='lbfgs', max_iter=5000)

        if debug:
            print(f"\nLikelihoodAgent train_from_logs() "
                  f"\nmodel.fit <- features size {len(features)}")
            
            for feature in features:
                print(f"\nmodel.fit <- feature size {len(feature)} : {feature}")

            print(f"\nmodel.fit <- rewards size {len(rewards)} \n{rewards}")

        self.model.fit(features, rewards) # ----- LEARN THE MODEL BY REWARDS ! X = FEATURES/VIEWS, Y = REWARDS ---> PROBABILITY OF REWARD




    def _score_products(self, user_state):

        all_action_features = np.array([
            self._create_features(user_state, action)
            for action in range(self.num_products)
        ])

        if debug:
            # print(f"\nLikelihoodAgent "
            #       f"\nall_action_features {all_action_features}"
            #       f"\n self.model.predict_proba {self.model.predict_proba(all_action_features)}")

            print(f"\nLikelihoodAgent _score_products() user_state {user_state} return {self.model.predict_proba(all_action_features)[:, 1]}")

        # predict_proba returns probability for 0 and 1 - [0.97692387 0.02307613]
        # Using [:,1] in the code will give you the probabilities of 1 (product view)

        return self.model.predict_proba(all_action_features)[:, 1]





    def act(self, observation, reward, done):

        """Act method returns an action based on current observation and past history"""

        if debug:
            print(f"\nLikelihoodAgent ACT() reward {reward} done {done} \nobservation {observation.sessions()}")

        self.feature_provider.observe(observation)

        user_state = self.feature_provider.features(observation)

        # prob is array of products click probability:
        # _score_products returns [0.09832074 0.02307613 0.01848091 0.02051842 0.04991053 0.0237508 0.01884811 0.01946998 0.02045175 0.02017838]

        prob = self._score_products(user_state)

        action = np.argmax(prob) # -> 0 (returns MAX FROM prob above)

        if debug:
            print(f"LikelihoodAgent ACT() scored products - prob {prob} action {action}")

        ps = 1.0
        all_ps = np.zeros(self.num_products)
        all_ps[action] = 1.0

        if debug:
            print(f"LikelihoodAgent ACT() user_state {user_state} prob {prob} action {action} ps {ps} ps-a {all_ps}")

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': ps,
                'ps-a': all_ps,
            }
        }

    def reset(self):

        self.feature_provider.reset()





class CountFeatureProvider(FeatureProvider):

    """Feature provider as an abstract class that defines interface of setting/getting features"""

    def __init__(self, config):

        super(CountFeatureProvider, self).__init__(config)

        self.feature_data = np.zeros((self.config.num_products))


    def observe(self, observation):

        """Consider an Organic Event for a particular user"""

        for session in observation.sessions():
            self.feature_data[int(session['v'])] += 1


    def features(self, observation):

        """Provide feature values adjusted to a particular feature set"""

        return self.feature_data


    def reset(self):

        self.feature_data = np.zeros((self.config.num_products))
