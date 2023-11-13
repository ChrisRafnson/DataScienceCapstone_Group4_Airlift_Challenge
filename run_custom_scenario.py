from airlift.envs import PlaneType
from airlift.envs.generators.cargo_generators import StaticCargoGenerator, DynamicCargoGenerator
from airlift.envs.airlift_env import AirliftEnv
from airlift.envs.events.event_interval_generator import EventIntervalGenerator
from airlift.envs.generators.airplane_generators import AirplaneGenerator
from airlift.envs.generators.airport_generators import RandomAirportGenerator
from airlift.envs.generators.route_generators import RouteByDistanceGenerator
from airlift.envs.generators.map_generators import PerlinMapGenerator
from airlift.envs.generators.world_generators import AirliftWorldGenerator
from airlift.envs.renderer import FlatRenderer
from airlift.solutions.solutions import doepisode

from solution.mysolution import MySolution
from solution.mysolution import ShortestPath

import pandas as pd

"""
Create an AirliftEnv using all the generators. There exist multiple generators for one thing. For example instead of using the
DynamicCargoGenerator we can also use the StaticCargoGenerator.
"""

"""
Uncomment the scenario below that you would like to use.
"""

# A simple scenario with no dynamic events
env = AirliftEnv(
    AirliftWorldGenerator(
        plane_types=[PlaneType(id=0, max_range=1.0, speed=0.05, max_weight=5)],
        airport_generator=RandomAirportGenerator(mapgen=PerlinMapGenerator(),
                                                 max_airports=3,
                                                 num_drop_off_airports=1,
                                                 num_pick_up_airports=1,
                                                 processing_time=1,
                                                 working_capacity=100,
                                                 airports_per_unit_area=2),
        route_generator=RouteByDistanceGenerator(route_ratio=1.25),
        cargo_generator=StaticCargoGenerator(num_of_tasks=1,
                                             soft_deadline_multiplier=10,
                                             hard_deadline_multiplier=20),
        airplane_generator=AirplaneGenerator(1),
    ),
    renderer=FlatRenderer(show_routes=True)
)

# # A more complicated scenario with dynamic events
# env = AirliftEnv( 
#     AirliftWorldGenerator(
#         plane_types=[PlaneType(id=0, max_range=1.0, speed=0.05, max_weight=5)],
#         airport_generator=RandomAirportGenerator(mapgen=PerlinMapGenerator(),
#                                                  max_airports=80,
#                                                  num_drop_off_airports=8,
#                                                  num_pick_up_airports=8,
#                                                  processing_time=8,
#                                                  working_capacity=2,
#                                                  airports_per_unit_area=8),
#         route_generator=RouteByDistanceGenerator(malfunction_generator=EventIntervalGenerator(1 / 300, 200, 300),
#                                                  route_ratio=5),
#         cargo_generator=DynamicCargoGenerator(cargo_creation_rate=1 / 100,
#                                               soft_deadline_multiplier=4,
#                                               hard_deadline_multiplier=12,
#                                               num_initial_tasks=40,
#                                               max_cargo_to_create=10),
#         airplane_generator=AirplaneGenerator(20),
#     ),
#     renderer=FlatRenderer(show_routes=True)
# )
filepath = ("master_database_updated.csv")
column_names = ['State', 'Action', 'Count', 'Value']
# master_df = pd.read_csv(filepath) 
master_df = pd.DataFrame(columns=column_names)
 
MySolution.episode_num = 1

iterations = 100 
my_solution = MySolution()
MySolution.updateReference(my_solution, master_df) #Update the policy functions database of state action pairs by passing it the current master dataframe

for i in range(iterations):

    env_info, metrics, time_taken, total_solution_time = \
    doepisode(env,
                solution=my_solution,
                render=False,
                render_sleep_time=0, # Set this to 0.1 to slow down the simulation
                env_seed=100,
                solution_seed=i)

    



    #Now that the episode is finished, we backtrack and update the score for all out state action pairs
    condition = my_solution.df['Value'] == "TBD" 
    my_solution.df.loc[condition, 'Value'] = -1 * metrics.score

    # Iterate through the current database
    # Assuming 'State' and 'Action' are the columns in both dataframes
    columns_to_match = ['State', 'Action']

    # Iterate through the current database 
    for index, row in my_solution.df.iterrows():
        # Extract the values from the current row
        values_to_check = [row[column] for column in columns_to_match]
        new_value = row['Value']

        # Check if there is a matching entry in the master database
        match = (master_df[columns_to_match] == values_to_check).all(axis=1)

        if match.any():
            # Update the existing entry in the master database
            master_df.loc[match, 'Value'] = ((master_df.loc[match, 'Value'] * master_df.loc[match, 'Count'])  + new_value) / (master_df.loc[match, 'Count'] + 1)
            master_df.loc[match, 'Count'] += 1
            
        else:
            # If no match found, add a new entry to the master database
            new_entry = dict(zip(columns_to_match, values_to_check))
            new_entry.update({'Count': 1, 'Value': new_value})

            # Create a DataFrame for the new entry
            new_entry_df = pd.DataFrame([new_entry])

            # Concatenate the new entry DataFrame with the master_df
            master_df = pd.concat([master_df, new_entry_df], ignore_index=True)


    

    MySolution.updateReference(my_solution, master_df) #Update the policy functions database of state action pairs by passing it the current master dataframe


    # print("Missed Deliveries: {}".format(metrics.missed_deliveries))
    # print("Lateness:          {}".format(metrics.total_lateness))
    # print("Total flight cost: {}".format(metrics.total_cost))
    print("Score:             {}".format(metrics.score) + "EPISODE NUMBER:" + str(i))




# Save the updated master database to a file
master_df.to_csv(filepath, index=False)
