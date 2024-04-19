import copy
import networkx as nx
from networkx import NetworkXNoPath
from ordered_set import OrderedSet

from airlift.envs.airlift_env import ObservationHelper as oh, ActionHelper, NOAIRPORT_ID
from airlift.solutions import Solution
from airlift.envs import ActionHelper


#These modules have been imported after the fact, they are necessary only for our solution
#The modules above are necessary for the simulator and the baseline solutions
import numpy as np
import pandas as pd
import random
import ast

class Q_learning(Solution):
    actions_returned = 0
    episode_num = 0 #initialize episode number
    learning_rate = .6
    discount_factor = .95
    epsilon = .30
    epsilon_decay_rate = .98
    Q_table = dict()

    previous_reduced_state = None
    last_action_taken = None
    current_reduced_state = None

    missed_deliveries_past = 0 #number of missed deliveries at the time of the last step

    def __init__(self):
        super().__init__()

    #This function just allows us to pass in the master dataframe to use as our reference dataframe
    def updateReference(self, newDict):
        self.Q_table = newDict

    def updateEnv(self, env):
        self.env = env

    def updateHyperParameters(self, newLearningRate, newDiscountRate, newEpsilon, newDecayRate):
        self.learning_rate = newLearningRate
        self.discount_factor = newDiscountRate
        self.epsilon = newEpsilon
        self.epsilon_decay_rate = newDecayRate

    
    def rewardFunction(self, previous_state, state):
        myString = str(state[0])
        total_value=0

        # if self.env.metrics.missed_deliveries > self.missed_deliveries_past:
        #     total_value =- 50
        # elif state[1] < previous_state[1]:
        #     total_value =+ 50
        # elif myString == "PlaneState.PROCESSING" and state[1] > 0: #If the plane is processing, return 5
        #     total_value =+ 10
        # elif myString == "PlaneState.PROCESSING": #If the plane is processing, return 5
        #     total_value =+ 1
        # if str(state) == str(previous_state): 
        #     total_value =- 25

        return 0

    #This was given by the challenge
    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    def update_Qval(self, state, action, state_prime, manual_reward=0): #Most of this code will be the same as the update code below
        # Bellman is defined as: Q(s,a) = (1 - learning_rate) * Q(s,a) + (learning_rate) * (reward + (discount_factor) * max Q(s', a))
        s = str(state) #This is the string representation of our reduced_state var from last step
        a = str(action) #This casts the action into a string
        s_prime = str(state_prime) #This should be a string representation of the state we just entered

        if s not in self.Q_table: #In this case, s is not in the Q-table, so we add it without any associated actions for now
            self.Q_table.update({s : {}})

        if a not in self.Q_table.get(s): #In this case, the action associate with s has not been seen yet, so we add it to the sub dict
            self.Q_table.get(s).update({a : 0})

        if s_prime not in self.Q_table: #In this case, s_prime is not in the Q-table, so we add it without any associated actions for now
            self.Q_table.update({s_prime : {}})

        
        Q_s_a = self.Q_table.get(s).get(a) #Value of Q(s,a)
        if manual_reward != 0:
            reward = manual_reward
        else:
            reward = self.rewardFunction(previous_state=state, state=state_prime) #The reward for getting to Q(s', a)
        if len(self.Q_table.get(s_prime).values()) > 0:
            maxQ_s_prime_a = max(self.Q_table.get(s_prime).values()) #The maximum value of all variations of Q(s', a')
        else:
            maxQ_s_prime_a = 0

        # Now we should be ready to finally use the bellman equation, which as a reminder is defined as:
        # Q(s,a) = (1 - learning_rate) * Q(s,a) + (learning_rate) * (reward + (discount_factor) * max Q(s', a))
        # As a reminder, for this version, we are rewarding the action rather than the resulting state.
        # We are also updating the original state, not the new one
        Bellman_solution = (1 - self.learning_rate) * (Q_s_a) + (self.learning_rate) * ((reward) + (self.discount_factor) * (maxQ_s_prime_a))
        self.Q_table.get(s).update({a : Bellman_solution}) #Update the value


    #The meat of the algorithm, this is where the decisions are made in theory.
    def policies(self, obs, dones, infos=0):

        #State Representation
        #=============================================================================================================================

        # get the complete state of the environment
        gs = self.get_state(obs)
        active_cargo = gs["active_cargo"]
        self.current_active_cargo = len(active_cargo)

        agent = gs["agents"]["a_0"] #grab the agent, there is only one currently

        # Most of the variables that we find useful for our state are contained within the "agent" class
        # this variable takes the values that we want from the "agent" class and stores them. This is
        # what is recorded in our state-action pair dataframe. This can be adjusted as we like
        self.current_reduced_state = [agent["state"], #current agent state
            agent["current_weight"], #current weight of agent
            # agent["max_weight"], #max weight of agent
            agent["available_routes"], #available routes
            agent["current_airport"]] #location of the agent
            # cargo_array] #array of available cargo

        if self.previous_reduced_state is None:
            self.previous_reduced_state = self.current_reduced_state

        reduced_state_string = str(self.current_reduced_state)
        previous_reduced_state_string = str(self.previous_reduced_state)
        
        #Decision Making
        #=============================================================================================================================

        # #Select a random action
        # action = self._action_helper.sample_valid_actions(obs) #We don't have a policy yet, so we'll just use a random agent for now

        # Q-learning algorithm

        # First on the agenda is updating our epsilon value, which decides whether we are exploiting or exploring
        if self.episode_num > 1:
            episode_epsilon = self.epsilon * self.epsilon_decay_rate ** self.episode_num
        else:
            episode_epsilon = 1.00

        random_val = np.random.rand()

        if (reduced_state_string not in self.Q_table) or (len(self.Q_table.get(reduced_state_string)) == 0) or (random_val < episode_epsilon):
            action = self._action_helper.sample_valid_actions(obs)
        else:
            best_action = max(self.Q_table.get(reduced_state_string), key=self.Q_table.get(reduced_state_string).get) #, key=self.Q_table.get(reduced_state_string).get)
            action = ast.literal_eval(best_action)

        self.last_action_taken = str(action)

        # Updating and Managing the Q-Table
        #============================================================================================================================
        # Keys are the state only, values are a dictionary which includes actions and their values.
        # However, this version waits until moving to the new state, updating the Q value for the
        # previous state. 
        #============================================================================================================================
        
        if self.actions_returned != 0:
            self.update_Qval(self.previous_reduced_state, self.last_action_taken, self.current_reduced_state)
        
        #End of this step, update relevant values
        self.previous_reduced_state = self.current_reduced_state
        self.actions_returned = self.actions_returned + 1
        self.num_active_cargo = self.current_active_cargo
        self.missed_deliveries_past = self.env.metrics.missed_deliveries
        return action

 