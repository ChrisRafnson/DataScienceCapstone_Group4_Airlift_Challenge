from airlift.envs import AirliftEnv, AirliftWorldGenerator, ActionHelper
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

# Agent algorithm goes here
def policy(obs):
    actions = ActionHelper.sample_valid_actions(observation=obs)
    return actions

env = AirliftEnv(AirliftWorldGenerator())
obs = env.reset()
while True:
    actions = policy(obs)
    obs, rewards, dones, infos = env.step(actions)
    env.render()
    if all(dones.values()):
        break