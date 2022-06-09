import gym

from rl_algorithm.common.policies import MlpPolicy
from rl_algorithm.common import make_vec_env
from rl_algorithm.ppo2 import PPO2
from carla_e2e import CarlaEnv
import argparse

def main(load_model,ep_length, target_v, seed = 7):

    # multiprocess environment
    env = CarlaEnv(ep_length, target_v)

    model = PPO2(MlpPolicy, env, verbose=0)
    model.learn(total_timesteps=250000)
    model.save("ppo2_carla")

    del model # remove to demonstrate saving and loading

    model = PPO2.load("ppo2_carla")

    # Enjoy trained agent
    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)


if __name__ == "__main__":
    params = {
        # screen size of cv2 window
        'obs_size': (160, 100),
        # time interval between two frames
        'dt': 0.025,
        # filter for defining ego vehicle
        'ego_vehicle_filter': 'vehicle.lincoln*',
        # CARLA service's port
        'port': 2000,
        # mode of the task, [random, roundabout (only for Town03)]
        #'task_mode': TASK_MODE,
        # mode of env (test/train)
        'code_mode': 'test',
        # maximum timesteps per episode
        'max_time_episode': 250,
        # desired speed (m/s)
        'desired_speed': 15,
        # maximum times to spawn ego vehicle
        'max_ego_spawn_times': 100,
    }
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--load', default=False, type=bool, help='whether to load existing model')
    #parser.add_argument('--map', type=str, default='Town03', help='name of carla map')
    parser.add_argument('--repeat-action', default=1, type=int, help='number of steps to repeat each action')
    parser.add_argument('--episode-length', default=400, type=int, help='maximum number of steps per episode')
    parser.add_argument('--target_v', default=6, type=int, help='desired velocity')
    parser.add_argument('--seed', type=int, default=7, help='random seed for initialization')

    args = parser.parse_args()
    load_model = args.load
    repeat_action = args.repeat_action
    steps_per_ep = args.episode_length
    ep_length = args.episode_length
    target_v = args.target_v
    seed = args.seed

    main(load_model,steps_per_ep, target_v)