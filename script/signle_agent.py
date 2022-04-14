# This is a sample Python script.

import argparse

try:
    import numpy as np
    import sys
    from os import path as osp
except ImportError:
    raise RuntimeError('import error!')


from queue import Empty
from carla_env.sim_world import WorldInit
from carla_env.sim_vehicle import VehicleInit


def main():

    argparser = argparse.ArgumentParser(description='Carla ArgParser practice')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server')
    argparser.add_argument('-p', '--port', default=2000, type=int, help='TCP port to listen to')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    args = argparser.parse_args()
    running = True

    try:
        env = WorldInit(args)
        ego_car = VehicleInit(env)

        while running:
            env.world.tick()
            env.spec_update()
            control_comd = ego_car.rule_based_step()
            env.ego_car.apply_control(control_comd)

            print("----------------------")

            try:
                env.sensor_queue.get()
            except Empty:
                print('some of the sensor information is missed')
    finally:
        env.__del__()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Exit by user')


