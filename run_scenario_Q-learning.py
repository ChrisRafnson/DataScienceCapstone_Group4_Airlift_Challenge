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

from solution.mysolution import ShortestPath
from solution.mysolution import Q_learning

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


#Initial value for episode number
Q_learning.episode_num = 1

iterations = 150 #This sets the number of episodes to run

# Selects the solution class to use and initializes the object.
# I created a few variables within the solution class that need to be initialized before it runs
my_solution = Q_learning()
my_solution.updateEnv(env)

# Run the episodes for the number of iterations specified
for i in range(iterations):

    env_info, metrics, time_taken, total_solution_time = \
    doepisode(env,
                solution=my_solution,
                render=False,
                render_sleep_time=0, # Set this to 0.1 to slow down the simulation
                env_seed=100,
                solution_seed=i)

    Q_learning.episode_num = Q_learning.episode_num + 1

    # print("Missed Deliveries: {}\n".format(metrics.missed_deliveries))
    # print("Lateness:          {}\n".format(metrics.total_lateness))
    # print("Total flight cost: {}\n".format(metrics.total_cost))
    print("Score:             {}\n".format(metrics.score) + "EPISODE NUMBER:" + str(i)) #prints out the score for the episode that just occured





