#!/usr/bin/env python3

'''
    The node for running Xarm OpenAI Gym environment 
'''

import gym
import time
import numpy
import random
from gym import wrappers
from stable_baselines3 import SAC

# ROS packages required
import rospy
import rospkg

# import our training environment
import Gym_xarm

config = {
    'GUI': True, 
    'reward_type':'dense_diff',
    'Sim2Real': False,
    'Digital_twin': False,
}

if __name__ == '__main__':

    rospy.init_node('run_xarm_gym', anonymous=True)

    # Create the Gym environment
    env = gym.make('Gym_xarm/XarmReachSac-v0', config = config, render_mode="rgb_array")
    rospy.loginfo ( "Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('gym_ros_pybullet')
    outdir = pkg_path + '/train'
    # env = wrappers.Monitor(env, outdir, force=True) 
    rospy.loginfo ( "Monitor Wrapper started")

    # Load the pre-trained RL model
    loaded_model = SAC.load(f"{outdir}/Best_model_sac_baseline")

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    running_steps = rospy.get_param("/running_steps")
    print('running_steps: ', running_steps)

    # Reset the environment
    obs = env.reset()

    for i in range(running_steps):
        action, _states = loaded_model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones or info.get("is_success", False):
            print("Success?", info.get("is_success", False))
            
            obs = env.reset()     
            time.sleep(0.5) 
    env.close()
    rospy.loginfo("Running Gym based on trained RL model done!")