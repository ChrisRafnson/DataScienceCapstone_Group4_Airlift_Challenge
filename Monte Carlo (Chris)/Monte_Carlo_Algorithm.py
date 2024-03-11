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

class Monte_Carlo_Method(Solution):
    actions_returned = 0
    episode_num = 0 #initialize episode number
    episode_buffer = [] #this will store all of the state action pairs and the cumulative rewards for the episode
    MC_table = {} #The table of the state action pairs, we need this to make decisions lol
    current_reduced_state = None
    previous_reduced_state = None
    last_action_taken = None
   
    def __init__(self):
        super().__init__()

    #This function just allows us to pass in the master dataframe to use as our reference dataframe
    def updateReference(self, newDict):
        self.MC_table = newDict

    def updateEnv(self, env):
        self.env = env
    
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

    def updateBuffer(self, reward):
        #Look at every current entry and add the reward into the running total
        for entry in self.episode_buffer:
            entry[2] = entry[2] + reward
        
    def updateTable(self,final_reward):
        #Before we can update the master table, we need to update the buffer one last time.
        self.updateBuffer(final_reward)


        #Note: State/Action pairs store both the value and the number of occurences
        for entry in self.episode_buffer:
            s = entry[0]
            a = entry[1]
            reward = entry[2]

            #We need to first ensure that the state/action already exists in the Table
            if s not in self.MC_table: 
                self.MC_table.update({s : {}})

            if a not in self.MC_table.get(s): 
                self.MC_table.get(s).update({a : [0, 1]})

            #Determine the new values
            old_entry = self.MC_table.get(s).get(a)
            old_val = old_entry[0]
            old_num_occurences = old_entry[1]

            new_num_occurences = old_num_occurences + 1
            new_val = ((old_val * old_num_occurences) + reward) / new_num_occurences

            #Put the new values back into the Table
            self.MC_table.get(s).update({a : [new_val, new_num_occurences]})

    def clearBuffer(self):
        self.episode_buffer = []

    #This was given by the challenge
    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

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

        # Monte Carlo Decision Making
        # Our policy will be to select the state-action with this highest value, like we did for Q-learning and Direct Evaluation

        # We will have an inital exploration phase of ten episodes, where the random agent is our policy and we record the values
        if self.episode_num < 10:
            action = self._action_helper.sample_valid_actions(obs)
        else: #After the ten episodes, we revert to the "greedy" policy
            if (reduced_state_string not in self.MC_table) or (len(self.MC_table.get(reduced_state_string)) == 0):
                action = self._action_helper.sample_valid_actions(obs)
            else:
                # best_action = max(self.MC_table.get(reduced_state_string), key=self.MC_table.get(reduced_state_string).get) #, key=self.Q_table.get(reduced_state_string).get)
                best_action = max(self.MC_table.get(reduced_state_string), key=lambda k: self.MC_table.get(reduced_state_string)[k][0])
                action = ast.literal_eval(best_action)

        self.last_action_taken = str(action)

        # Updating and Managing the Episode Buffer
        #============================================================================================================================
        # The buffer is a list since we are performing an every look MC method, it used to be a dictionary for first look method
        #============================================================================================================================

        #Firstly, get the reward for this time step:
        reward = self.rewardFunction(self.previous_reduced_state, self.current_reduced_state)

        #Check if this state and action is in the table:
        s = reduced_state_string #current state this timestep
        a = self.last_action_taken #action taken this timestep

        #It does not matter whether or not our state or action has been seen before since the buffer records everything

        buffer_entry = [s, a, 0]

        self.episode_buffer.append(buffer_entry)

        self.updateBuffer(reward)

        #End of this step, update relevant values
        self.previous_reduced_state = self.current_reduced_state
        self.actions_returned = self.actions_returned + 1
        return action