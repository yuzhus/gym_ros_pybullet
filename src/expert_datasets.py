#!/usr/bin/env python3

'''
    The code for generating expert demonstrations
'''
import gym
import numpy as np
from gym import spaces
import time
import numpy
import random
from gym import wrappers
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl

# ROS packages required
import rospy
import rospkg

# import our training environment
import Gym_xarm

config = {
    'GUI': True, 
    'reward_type':'dense_diff',
    'Sim2Real': False,
    'Digital_twin': True
}

if __name__ == '__main__':

    rospy.init_node('expert_node', anonymous=True)

    # Create the Gym environment
    env = gym.make('Gym_xarm/XarmReachSac-v0', config = config, render_mode="rgb_array")
    rospy.loginfo ( "Gym environment done")

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('gym_ros_pybullet')
    outdir = pkg_path + '/train'
    # env = wrappers.Monitor(env, outdir, force=True) 
    rospy.loginfo ( "Monitor Wrapper started")

    loaded_model = SAC.load(f"{outdir}/Best_expert_sofar")

    # Initialize the custom replay buffer 465 transitions
    buffer_size = 330
    num_episodes = 10
    action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
    observation_space = spaces.Dict({
        "observation": spaces.Box(-np.inf, np.inf, shape=(18,), dtype='float32'),
        "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
        "obstacles": spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
    })
    demo_replay_buffer = DictReplayBuffer(buffer_size, observation_space, action_space, handle_timeout_termination = False) # 60% sampled from human demonstrations
    print(f"Initialized demo_replay_buffer: {type(demo_replay_buffer)}")

    # Start a new episode
    for _ in range(num_episodes):
        obs = env.reset()
        for i in range(50):
            action, _states = loaded_model.predict(obs, deterministic=True)
            next_obs, reward, done, info = env.step(action)
            env.render()
            demo_replay_buffer.add(obs, next_obs, action, reward, done, infos=info)
            print("demoreplaybuffer",demo_replay_buffer.pos)
            obs = next_obs

            if done or info.get("is_success", False):
                print("Success?", info.get("is_success", False))
                print("Safe?", info.get("is_safe", False))
                
                time.sleep(1./240)
                obs = env.reset()
                break      
    env.close()

    save_to_pkl(path=f"{outdir}/Expert_demo_best", obj=demo_replay_buffer)
    print(f"The loaded_model has {demo_replay_buffer.size()} transitions in its buffer")

    rospy.loginfo("Collecting expert demonstrations done!")
