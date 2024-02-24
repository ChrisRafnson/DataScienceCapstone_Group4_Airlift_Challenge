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

class Direct_Evaluation(Solution):
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    episode_num = 0 #initialize episode number
    column_names = ['State', 'Action', 'Count','Sum', 'Average']
    df = pd.DataFrame() #This is the temporary dataframe to record the state-action pairs only for this episode
    reference_df = pd.DataFrame(columns=column_names) #This is our "model", what we are using and referencing to make decisions.

    def __init__(self):
        super().__init__()

    #This function just allows us to pass in the master dataframe to use as our reference dataframe
    def updateReference(self, newDataFrame):
        self.reference_df = newDataFrame

    #This was given by the challenge
    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        # Currently, the evaluator will NOT pass in an observation space or action space (they will be set to None)
        super().reset(obs, observation_spaces, action_spaces, seed)

        # Create an action helper using our random number generator
        self._action_helper = ActionHelper(self._np_random)

    #The meat of the algorithm, this is where the decisions are made in theory.
    def policies(self, obs, dones):

        #State Representation
        #=============================================================================================================================

        # get the complete state of the environment
        gs = self.get_state(obs)
        
        # The following snippet grabs the necessary cargo information for our state, this will be useful later

        # cargo_array = [] #Get the necessary values from each cargo entry
        # active_cargo = gs["active_cargo"]
        # for cargo in active_cargo:
        #     cargo_array.append([
        #         cargo[0],
        #         cargo[1],
        #         cargo[2],
        #         cargo[3] 
        #     ])

        agent = gs["agents"]["a_0"] #grab the agent, there is only one currently

        # Most of the variables that we find useful for our state are contained within the "agent" class
        # this variable takes the values that we want from the "agent" class and stores them. This is
        # what is recorded in our state-action pair dataframe. This can be adjusted as we like
        reduced_state = [agent["state"], #current agent state
            # agent["current_weight"], #current weight of agent
            # agent["max_weight"], #max weight of agent
            agent["available_routes"], #available routes
            agent["current_airport"]] #location of the agent
            # cargo_array] #array of available cargo
        
        #Decision Making
        #=============================================================================================================================


        #Direct Evaluation Policy - This is what we're having trouble with

        # action = self._action_helper.sample_valid_actions(obs) # grab a random action in case we can't find a valid action

        # #Select the best action

        # maxval = -999999999
        # for index, row in self.reference_df.iterrows(): #Search the database for a match
        #     if row['State'] == reduced_state: #If our state has been encountered before, then we find the best action from that state
        #         if row['Value'] >= maxval:
        #             action = row['Action']
        #             maxval = row['Value']    
             
        #This is the random policy agent, works just fine

        #Select a random action
        action = self._action_helper.sample_valid_actions(obs) #We don't have a policy yet, so we'll just use a random agent for now

        #=============================================================================================================================

        #Create a new row for this state and action
        new_df_entry = pd.DataFrame({'State': [reduced_state], 'Action': [action], 'Episode': "TBD", 'Value': "TBD"})

        #add the new entry into our current dataframe
        self.df = pd.concat([self.df, new_df_entry], ignore_index=True)

        return action