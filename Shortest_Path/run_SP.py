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

import json
import csv
import os

from Shortest_Path_Algorithm import ShortestPath as SP


"""
Create an AirliftEnv using all the generators. There exist multiple generators for one thing. For example instead of using the
DynamicCargoGenerator we can also use the StaticCargoGenerator.
"""

"""
Uncomment the scenario below that you would like to use.
"""

# # A simple scenario with no dynamic events
env1 = AirliftEnv(
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

# # A more complex scenario with no dynamic events
env2 = AirliftEnv(
    AirliftWorldGenerator(
        plane_types=[PlaneType(id=0, max_range=1.0, speed=0.05, max_weight=5)],
        airport_generator=RandomAirportGenerator(mapgen=PerlinMapGenerator(),
                                                 max_airports=5,
                                                 num_drop_off_airports=1,
                                                 num_pick_up_airports=2,
                                                 processing_time=1,
                                                 working_capacity=100,
                                                 airports_per_unit_area=2),
        route_generator=RouteByDistanceGenerator(route_ratio=1.25),
        cargo_generator=StaticCargoGenerator(num_of_tasks=3,
                                             soft_deadline_multiplier=10,
                                             hard_deadline_multiplier=20),
        airplane_generator=AirplaneGenerator(1),
    ),
    renderer=FlatRenderer(show_routes=True)
)

# # A more complex scenario with no dynamic events
env3 = AirliftEnv(
    AirliftWorldGenerator(
        plane_types=[PlaneType(id=0, max_range=1.0, speed=0.05, max_weight=5)],
        airport_generator=RandomAirportGenerator(mapgen=PerlinMapGenerator(),
                                                 max_airports=10,
                                                 num_drop_off_airports=2,
                                                 num_pick_up_airports=3,
                                                 processing_time=1,
                                                 working_capacity=100,
                                                 airports_per_unit_area=2),
        route_generator=RouteByDistanceGenerator(route_ratio=1.25),
        cargo_generator=StaticCargoGenerator(num_of_tasks=8,
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



# Configuration parameters
version = "v1"  # You can change this to reflect the version of the simulation
run = "01"
env_config = "env3"  # Environment configuration used

# Directory setup
base_dir = os.path.join("Shortest_Path/simulation_results", version)
os.makedirs(base_dir, exist_ok=True)  # Create the directory if it doesn't exist


# Selects the solution class to use and initializes the object.
# I created a few variables within the solution class that need to be initialized before it runs
my_solution = SP()

# my_solution.updateEnv(env3)


# #Initial value for episode number
# my_solution.episode_num = 0

# CSV file setup
filename = f"{base_dir}/env_{env_config}_run_{run}.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Define the headers based on your requirements
    writer.writerow(["Episode Number", "Score", "Epsilon Value", "Number of Actions", "Missed Deliveries"])

iterations = 150  #This sets the number of episodes to run
# Run the episodes for the number of iterations specified
for i in range(iterations): 

    env_info, metrics, time_taken, total_solution_time = \
    doepisode(env3,
                solution=my_solution,
                render=False,
                render_sleep_time=0, # Set this to 0.1 to slow down the simulation
                env_seed=100,
                solution_seed=i)

    # my_solution.episode_num = my_solution.episode_num + 1

    # # Factor the episode score into the algorithm as one final reward
    # if metrics.missed_deliveries > 0:
    #     reward = -1 * metrics.score ** 2
    # else:
    #     if metrics.score < 0:
    #         reward = (1/metrics.score) * 1000
    #     else:
    #         reward = metrics.score * 10000


    # my_solution.updateTable(reward)
    # my_solution.clearBuffer()

    # Append results to the CSV file
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i, metrics.score, metrics.missed_deliveries])
    
    print("EPISODE NUMBER: {}".format(i))
    print("         Score: {}".format(metrics.score)) #prints out the score for the episode that just occured
    # print(" Epsilon Value: {}".format(my_solution.episode_epsilon)) #prints out the score for the episode that just occured
    # print(" Last Action Returned: {}".format(my_solution.last_action_taken))
    # print(" Number of Actions Returned: {}".format(my_solution.actions_returned))
    print("=============================================================================================")

    my_solution.actions_returned = 0


# json_data = json.dumps(my_solution.MC_table, indent=4)
# json_filename = os.path.join(base_dir, f'{env_config}_run_{run}_MC_table.json')
# with open(json_filename, mode='w') as fp:
#     fp.write(json_data)






