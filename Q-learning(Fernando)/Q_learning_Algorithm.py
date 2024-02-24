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
    """
    Utilizing this class for your solution is required for your submission. The primary solution algorithm will go inside the
    policy function.
    """
    actions_returned = 0
    episode_num = 0 #initialize episode number
    learning_rate = .7
    discount_factor = .95
    epsilon = .30
    epsilon_decay_rate = .98
    Q_table = dict()

    previous_reduced_state = None
    last_action_taken = None
    current_reduced_state = None

    # current_active_cargo = 0 #The number of active cargo in this step
    # num_active_cargo = 0 #The number of active cardo in the last step

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
        
        return total_value

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
    def policies(self, obs, dones):

        #State Representation
        #=============================================================================================================================

        # get the complete state of the environment
        gs = self.get_state(obs)
        active_cargo = gs["active_cargo"]
        self.current_active_cargo = len(active_cargo)

        #create a copy of the environment to play with
        environment_copy = copy.deepcopy(self.env)
        
        # The following snippet grabs the necessary cargo information for our state, this will be useful later

        # cargo_array = [] #Get the necessary values from each cargo entry
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
        if self.episode_num > 10:
            episode_epsilon = self.epsilon * self.epsilon_decay_rate ** self.episode_num
        else:
            episode_epsilon = 1.00

        if (reduced_state_string not in self.Q_table) or (len(self.Q_table.get(reduced_state_string)) == 0) or (np.random.rand() < episode_epsilon):
            action = self._action_helper.sample_valid_actions(obs)
        else:
            best_action = max(self.Q_table.get(reduced_state_string)) #, key=self.Q_table.get(reduced_state_string).get)
            action = ast.literal_eval(best_action)

        self.last_action_taken = str(action)

        # Updating and Managing the Q-Table
        #=============================================================================================================================


        # Option 1 : Keys are the combination of state and action, makes finding the maximum Q-value of the resulting state
        # difficult to find as there is no good way to generate valid actions without using the random samplers

        #--------------------------------------------------------------------------------------------------------------------------------------
        
        # combined_state_action = str(reduced_state) + "_" + str(action)
        # print(combined_state_action)

        # if combined_state_action not in self.Q_table: #If the Q state is not in the table, then we add it with an initial value of zero
        #     self.Q_table.update({combined_state_action : 0})
        # else: #Otherwise we need to update the Q-value using the bellman equation
            
        #     #Bellman is defined as: Q(s,a) = (1 - learning_rate) * Q(s,a) + (learning_rate) * (reward + (discount_factor) * max Q(s', a))

        #     current_Q_value = self.Q_table.get(combined_state_action)
        #     reward = self.rewardFunction(reduced_state)

        #     # The final component we need for Bellman is the maximum value of the resulting states, this allows us to assume
        #     # optimal play after we take our action

        #     actions = self.env.action_space("a_0")

        #----------------------------------------------------------------------------------------------------------------------------------------

        #Option 2 : Keys are the state only, values are a dictionary which includes actions and their values.

        #----------------------------------------------------------------------------------------------------------------------------------------

        
        # action_string = str(action)

        # if reduced_state_string not in self.Q_table: #If the Q state is not in the table, then we add it with a dictionary that associates that action with a value of 0
        #     self.Q_table.update({reduced_state_string : {action_string : 0}})
            
        # elif action_string not in self.Q_table.get(reduced_state_string): #In this case the state is in the Q_table but not the action, so we add it to the sub dictionary
        #     self.Q_table.get(reduced_state_string).update({action_string : 0})
            
        # # No matter what we will need to update the Q-value using the bellman equation

        # # Bellman is defined as: Q(s,a) = (1 - learning_rate) * Q(s,a) + (learning_rate) * (reward + (discount_factor) * max Q(s', a))

        # current_Q_value = self.Q_table.get(reduced_state_string).get(action_string) # Grab the value of the associated state and action from the nested dictionaries
        # reward = self.rewardFunction(reduced_state) #Grab the reward from our reward function

        # # The last component of the Bellman equation is the maximum possible value from the resulting state.
        # # To do this we need to step forward in the simulation to figure out the resulting state. My current theory as to how to implement this without
        # # affecting the "real" environment is to create a copy of the environment and then call step on that environment. We can create an copy at the beginning of every
        # # call of this function, getting the most updated picture of the environment.

        # new_obs, new_rewards, new_dones, _ = environment_copy.step(action) #Execute the action within our copy of the environment

        # new_state = self.get_state(new_obs) #Get the state of the new observation


        # # Now we can copy the code from above and perform the same data manipulation as earlier to achieve 
        # # The new state after taking the selected action. 
        # new_agent = new_state["agents"]["a_0"] #grab the agent, there is only one currently

        # new_reduced_state = [new_agent["state"], #current agent state
        #     # agent["current_weight"], #current weight of agent
        #     # agent["max_weight"], #max weight of agent
        #     new_agent["available_routes"], #available routes
        #     new_agent["current_airport"]] #location of the agent
        #     # cargo_array] #array of available cargo

        # new_reduced_state_string = str(new_reduced_state) #Convert the new reduced state to a string to search the dictionary

        # if new_reduced_state_string in self.Q_table: #If the new state is in the table then just find the max values, otherwise the value is 0
        #     max_value_resulting_state = max(self.Q_table.get(new_reduced_state_string).values())
        # else:
        #     max_value_resulting_state = 0

        # # Now we should be ready to finally use the bellman equation, which as a reminder is defined as:
        # # Q(s,a) = (1 - learning_rate) * Q(s,a) + (learning_rate) * (reward + (discount_factor) * max Q(s', a))
        # # As a reminder, for this version, we are rewarding the action rather than the resulting state.
        # # We are also updating the original state, not the new one

        # Bellman_solution = (1 - self.learning_rate) * (current_Q_value) + (self.learning_rate) * ((reward) + (self.discount_factor) * (max_value_resulting_state))
        # self.Q_table.get(reduced_state_string).update({action_string : Bellman_solution})


        #----------------------------------------------------------------------------------------------------------------------------------------

        # Option 3 : Keys are the state only, values are a dictionary which includes actions and their values.
        #               However, this version waits until moving to the new state, updating the Q value for the
        #               previous state. No deep copies of the environment of simulation required.
        #               Additionally, the code for updating Q_values has been condensed, Option 2 will
        #               likely be retired with this new option. The function is Update_Qval().

        #----------------------------------------------------------------------------------------------------------------------------------------
        
        if self.actions_returned != 0:
            self.update_Qval(self.previous_reduced_state, self.last_action_taken, self.current_reduced_state)
        
        self.previous_reduced_state = self.current_reduced_state
        self.actions_returned = self.actions_returned + 1
        self.num_active_cargo = self.current_active_cargo
        self.missed_deliveries_past = self.env.metrics.missed_deliveries
        return action

 