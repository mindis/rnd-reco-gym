from abc import ABC

import gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete
from numpy.random.mtrand import RandomState
from scipy.special import expit as sigmoid
from tqdm import trange

from .configuration import Configuration
from .context import DefaultContext
from .features.time import DefaultTimeGenerator
from .observation import Observation
from .session import OrganicSessions
from ..agents import Agent

# Arguments shared between all environments.

env_args = {
    'num_products': 1000,
    'num_users': 1000,
    'random_seed': np.random.randint(2 ** 31 - 1),
    # Markov State Transition Probabilities.

    'prob_leave_bandit': 0.01,    # STOP PROBABILITY, 0.01 - DEFAULT
    'prob_leave_organic': 0.01,    # STOP PROBABILITY, 0.01 - DEFAULT

    'prob_bandit_to_organic': 0.05,
    'prob_organic_to_bandit': 0.25,
    
    'normalize_beta': False,
    'with_ps_all': False
}

debug = False

className = "envs/abstract"

# Static function for squashing values between 0 and 1.
def f(mat, offset=5):
    """Monotonic increasing function as described in toy.pdf."""
    return sigmoid(mat - offset)


# Magic numbers for Markov states.
organic = 0
bandit = 1
stop = 2

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

class AbstractEnv(gym.Env, ABC):

    def __init__(self):
        gym.Env.__init__(self)
        ABC.__init__(self)

        self.first_step = True
        self.config = None
        self.state = None
        self.current_user_id = None
        self.current_time = None
        self.empty_sessions = OrganicSessions()



    def reset_random_seed(self, epoch=0):
        # Initialize Random State.
        assert (self.config.random_seed is not None)
        self.rng = RandomState(self.config.random_seed + epoch)



    def init_gym(self, args):

        self.config = Configuration(args)

        # Defining Action Space.
        self.action_space = Discrete(self.config.num_products)

        if 'time_generator' not in args:
            self.time_generator = DefaultTimeGenerator(self.config)
        else:
            self.time_generator = self.config.time_generator

        # Setting random seed for the first time.
        self.reset_random_seed()

        if 'agent' not in args:
            self.agent = None
        else:
            self.agent = self.config.agent

        # Setting any static parameters such as transition probabilities.
        self.set_static_params()

        # Set random seed for second time, ensures multiple epochs possible.
        self.reset_random_seed()




    def reset(self, user_id=0):
        # Current state.
        self.first_step = True
        self.state = organic  # Manually set first state as Organic.

        self.time_generator.reset()
        if self.agent:
            self.agent.reset()

        self.current_time = self.time_generator.new_time()
        self.current_user_id = user_id

        # Record number of times each product seen for static policy calculation.
        self.organic_views = np.zeros(self.config.num_products)




    def generate_organic_sessions(self): # -- RUN CHAIN OF ORGANIC VIEWS !!!

        if debug:
            print(f"   {className} generate_organic_sessions () state {self.state}")

        # Initialize session.
        session = OrganicSessions()

        while self.state == organic: # -- UPDATE ORGANIC VIEWS !!!

            # Add next product view.
            self.update_product_view()

            session.next(
                DefaultContext(self.current_time, self.current_user_id),
                self.product_view
            )

            # Update markov state.
            self.update_state()

        return session







    def step(self, action_id):
        """

        Parameters
        ----------
        action_id : int between 1 and num_products indicating which product recommended (aka which ad shown)

        Returns
        -------
        observation, reward, done, info : tuple

            observation (tuple) :

                a tuple of values (is_organic, product_view)

                is_organic - True  if Markov state is `organic`,
                             False if Markov state `bandit` or `stop`.

                product_view - if Markov state is `organic` then it is an int
                               between 1 and P where P is the number of
                               products otherwise it is None.

            reward (float) :

                if the previous state was

                    `bandit` - then reward is 1 if the user clicked on the ad
                               you recommended otherwise 0

                    `organic` - then reward is None

            done (bool) :

                whether it's time to reset the environment again.
                An episode is over at the end of a user's timeline (all of
                their organic and bandit sessions)

            info (dict) :
                 this is unused, it's always an empty dict
        """

        # No information to return.
        info = {}

        if debug:
            print(f".. {className} step(action {action_id}) state {self.state}")

        if self.first_step:
            assert (action_id is None)
            self.first_step = False

            if debug:
                print(f".. {className} step() FIRST STEP => RUN generate_organic_sessions()")

            sessions = self.generate_organic_sessions() # -- RUN CHAIN OF ORGANIC VIEWS

            return (
                Observation(
                    DefaultContext(
                        self.current_time,
                        self.current_user_id
                    ),
                    sessions
                ),
                None,
                self.state == stop,
                info
            )

        assert (action_id is not None)

        # Calculate reward from action.

        reward = self.draw_click(action_id) # reward = click !

        if debug:
            print(f".. {className} step() state {self.state} reward {reward} from action_id {action_id}")

        self.update_state()

        if reward == 1:
            self.state = organic  # After a click, Organic Events always follow.

        # Markov state dependent logic.

        if debug:
            print(f".. {className} step () state {self.state}")

        if self.state == organic:

            if debug:
                print(f".. {className} step () RUN generate_organic_sessions()")

            sessions = self.generate_organic_sessions() # -- RUN CHAIN OF ORGANIC VIEWS

        else:
            sessions = self.empty_sessions

        if debug:
            print(f".. {className} step () state {self.state}")

        return (
            Observation(
                DefaultContext(self.current_time, self.current_user_id),
                sessions
            ),
            reward,
            self.state == stop,
            info
        )









    def step_offline(self, observation, reward, done):

        """Call step function wih the policy implemented by a particular Agent."""

        if debug:
            print(f"\n{className} step_offline() state {self.state} done {done} reward {reward} products number {self.config.num_products}")

        if self.first_step:
            action = None

        else:
            assert (hasattr(self, 'agent'))
            assert (observation is not None)

            if debug:
                print(f"{className} step_offline() state {self.state} agent {self.agent} reward {reward}\nobservation {observation.sessions()}")

            #------------------ AGENT EXISTS --------------------

            if self.agent:

                if debug:
                    print(f"{className} step_offline() state {self.state} RUN agent ACT")

                action = self.agent.act(observation, reward, done)                            # ---- RUN AGENT ACT

                if debug:
                    print(f"{className} step_offline() state {self.state} recommended action = {action}")

            #------------------ RANDOM ACTION --------------------

            else:

                # Select a Product randomly.

                if debug:
                    print(f"{className} step_offline() state {self.state} <-- SELECT RANDOM ACTION -->")

                action = {
                    't': observation.context().time(),
                    'u': observation.context().user(),

                    'a': np.int16(self.rng.choice(self.config.num_products)),
                    'ps': 1.0 / self.config.num_products,
                    'ps-a': (
                        np.ones(self.config.num_products) / self.config.num_products
                        if self.config.with_ps_all else
                        ()
                    ),
                }
                if debug:
                    print(f"{className} step_offline() state {self.state} recommended -RANDOM- action = {action}")

        if done:

            return (
                action,
                Observation(
                    DefaultContext(self.current_time, self.current_user_id),
                    self.empty_sessions
                ),
                0,
                done,
                None
            )

        else:

            if debug:
                print(f"{className} step_offline() state {self.state} run STEP (action {action['a']})")

            observation, reward, done, info = self.step(
                action['a'] if action is not None else None
            )

            if debug:
                print(f"{className} step_offline() state {self.state} return: action {action} done {done}\n"
                      f"observation {observation.sessions()}\n")

            return action, observation, reward, done, info






    # ------------------------------------------------------------------------------------
    #  GENERATE LOGS
    # ------------------------------------------------------------------------------------

    def generate_logs(

            self,
            num_offline_users: int,
            agent: Agent = None,
            num_organic_offline_users: int = 0
    ):
        """
        Produce logs of applying an Agent in the Environment for the specified amount of Users.
        If the Agent is not provided, then the default Agent is used that randomly selects an Action.
        """

        # print(f"{className} generate_logs START() state {self.state} agent {self.agent}")

        print('# ------------------------------------------------------------------------------------')
        print(f"#  GENERATE LOGS FOR AGENT {agent}")
        print('# ------------------------------------------------------------------------------------')

        if agent:
            old_agent = self.agent
            self.agent = agent

        data = {
            't': [],
            'u': [],
            'z': [],
            'v': [],
            'a': [],
            'c': [],
            'ps': [],
            'ps-a': [],
        }

        def _store_organic(observation):

            assert (observation is not None)
            assert (observation.sessions() is not None)

            for session in observation.sessions():
                data['t'].append(session['t'])
                data['u'].append(session['u'])
                data['z'].append('organic')                                     # Z = ORGANIC
                data['v'].append(session['v'])                                  # ORGANIC ONLY
                data['a'].append(None)
                data['c'].append(None)
                data['ps'].append(None)
                data['ps-a'].append(None)

        def _store_bandit(action, reward):

            if action:
                assert (reward is not None)

                data['t'].append(action['t'])
                data['u'].append(action['u'])
                data['z'].append('bandit')                                      # Z = BANDIT
                data['v'].append(None)                                          # NONE FOR BANDIT
                data['a'].append(action['a'])                                   # BANDIT ONLY
                data['c'].append(reward)                                        # REWARD - BANDIT ONLY
                data['ps'].append(action['ps'])                                 # BANDIT ONLY
                data['ps-a'].append(action['ps-a'] if 'ps-a' in action else ()) # BANDIT ONLY

        unique_user_id = 0

        if debug:

            print(f"\n{className} generate_logs() state {self.state} agent {agent}"
                  f"num_offline_users {num_offline_users} "
                  f"num_organic_offline_users {num_organic_offline_users}")

        for _ in trange(num_organic_offline_users, desc='Organic Users'):

            self.reset(unique_user_id)
            unique_user_id += 1

            observation, _, _, _ = self.step(None)                                                   # ONLINE STEP, NO ACTION

            _store_organic(observation)


        for _ in trange(num_offline_users, desc='Users'):

            self.reset(unique_user_id)
            unique_user_id += 1

            observation, reward, done, _ = self.step(None)

            while not done:

                _store_organic(observation)

                action, observation, reward, done, _ = self.step_offline(observation, reward, done)   # OFFLINE STEP -> RUN ACT

                _store_bandit(action, reward)


            _store_organic(observation)

            action, _, reward, done, _ = self.step_offline(observation, reward, done)   # OFFLINE STEP -> RUN ACT

            assert done, 'Done must not be changed!'

            _store_bandit(action, reward)

        data['t'] = np.array(data['t'], dtype=np.float32)
        data['u'] = pd.array(data['u'], dtype=pd.UInt16Dtype())
        data['v'] = pd.array(data['v'], dtype=pd.UInt16Dtype())
        data['a'] = pd.array(data['a'], dtype=pd.UInt16Dtype())
        data['c'] = np.array(data['c'], dtype=np.float32)

        if agent:

            self.agent = old_agent

        result = pd.DataFrame().from_dict(data)

        # if debug:
        #     print(f"\n{className} generate_logs() state {self.state} RETURN DATA \n{result}\n")

        return result
