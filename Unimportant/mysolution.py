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

class ShortestPath(Solution):
    def __init__(self):
        super().__init__()

        self.cargo_assignments = None
        self.path = None
        self.whole_path = None

    def reset(self, obs, observation_spaces=None, action_spaces=None, seed=None):
        super().reset(obs, observation_spaces, action_spaces, seed)
        state = self.get_state(obs)

        self.cargo_assignments = {a: None for a in self.agents}
        self.path = {a: None for a in self.agents}
        self.whole_path = {a: None for a in self.agents}
        self.multidigraph = oh.get_multidigraph(state)

        self._full_delivery_paths = {}

    def policies(self, obs, dones):
        state = self.get_state(obs)

        # Active cargo list should not have any delivered cargo
        assert all(c.location != c.destination for c in state["active_cargo"])

        # Since shortest paths calculation take most of the time, let's keep track of delivery paths, since these won't change.
        # We need to update this list with any dynamic cargo that appears.
        for c in state["active_cargo"]:
            if c.location != NOAIRPORT_ID and c.id not in self._full_delivery_paths:
                self._full_delivery_paths[c.id] = nx.shortest_path(self.multidigraph, c.location, c.destination, weight="cost")[1:]

        # Cargo needing to be delivered that is not assigned yet (sorted to make it deterministic)
        pending_cargo = [c for c in state["active_cargo"] if c.id not in self.cargo_assignments.values() and c.is_available == 1]
        actions = {a: None for a in self.agents}

        for a in self.agents:
            # If the agent is done, stop issuing actions for it

            if dones[a]:
                continue

            # If the airplane has a cargo assignment...
            if self.cargo_assignments[a] is not None:
                # Has it been delivered?
                if self.cargo_assignments[a] not in [c.id for c in state["active_cargo"]]:
                    # Unassign it
                    self.cargo_assignments[a] = None

            # If the airplane needs a new assignment...
            if pending_cargo and self.cargo_assignments[a] is None:
                # Check if there is any cargo needing to be delivered that is not assigned yet
                cargo_info = pending_cargo[self._np_random.choice(range(len(pending_cargo)))]

                if cargo_info.location != NOAIRPORT_ID:
                    full_delivery_path = self._full_delivery_paths[cargo_info.id]
                    try:
                        # Check if we should pick up this cargo...
                        # Can we make any progress after pickup? If not, move on to next cargo...
                        if not self.multidigraph.has_edge(cargo_info.location, full_delivery_path[0], obs[a]['plane_type']):
                            continue

                        # Generate a pickup path. If we can't reach the cargo, this will throw an exception
                        path = oh.get_lowest_cost_path(state, obs[a]["current_airport"],
                                                               cargo_info.location,
                                                               obs[a]["plane_type"])

                        # If we made it here, we should pick this cargo up. Complete the path to the dropoff location...
                        # Follow the full delivery path until we can't go further.
                        while full_delivery_path and self.multidigraph.has_edge(path[-1], full_delivery_path[0], obs[a]['plane_type']):
                            path.append(full_delivery_path.pop(0))

                        # Make the assignment
                        self.path[a] = path
                        self.cargo_assignments[a] = cargo_info.id
                        pending_cargo.remove(cargo_info)

                        # Once we have found a cargo and assigned it, we can break out of this loop.
                        break

                    except NetworkXNoPath as e:
                        # If there is no path, to pick up and/or deliver, don't complete the assignment
                        pass

            # If the plane is idle, assign a new action
            # We only assign an action while idle. If we try to assign a new action while the plane is not idle, we could hit on some glitches
            if oh.is_airplane_idle(obs[a]):
                actions[a] = {"process": 1,
                              "cargo_to_load": set(),
                              "cargo_to_unload": set(),
                              "destination": NOAIRPORT_ID}

                # If we have a path to follow, set next destination and takeoff when ready
                if self.path[a]:
                    next_destination = self.path[a][0]

                    # Have we arrived at the next destination? Pop that one off and set the next one in the list.
                    if obs[a]["current_airport"] == next_destination:
                        self.path[a].pop(0)
                        if self.path[a]:
                            next_destination = self.path[a][0]
                        else:
                            next_destination = NOAIRPORT_ID

                    actions[a]["destination"] = next_destination

                # Get info about the currently assigned cargo
                ca = oh.get_active_cargo_info(state, self.cargo_assignments[a])
                # If cargo is assigned

                if ca is not None:
                    if ca.id in obs[a]["cargo_onboard"]:
                        # If you're at final destination unload cargo or you have reached the end of your possible paths
                        if ca.destination == obs[a]['current_airport'] or not self.path[a]:
                            actions[a]["cargo_to_unload"].add(ca.id)
                            self.cargo_assignments[a] = None
                    elif ca.id in obs[a]['cargo_at_current_airport']:
                        actions[a]["cargo_to_load"].add(ca.id)

        return actions