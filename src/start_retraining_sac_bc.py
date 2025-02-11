#!/usr/bin/env python3

'''
    The training node for OpenAI Gym environment
'''
import os
import gym
import time
import random
import numpy as np
from gym import wrappers
import matplotlib.pyplot as plt

# SB3 packages required
from stable_baselines3 import SACBC, DASACBC
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import ReplayBuffer, DictReplayBuffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from gym import error, spaces, utils
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
# from stable_baselines3.common.vec_env import DummyVecEnv

# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import PoseStamped

# import our custom training environment
import Gym_xarm

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "Best_retrain_model_with_demo_v3")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True

class XarmEnvCollectData:
    def __init__(self, env, initial_position, demo_replay_buffer: Type[DictReplayBuffer], num_episodes=10):
        self.env = env
        self.demo_replay_buffer = demo_replay_buffer
        self.num_episodes = num_episodes
        self.prev_position = np.array(initial_position)
        self.relative_positions = []
        self.gripper_action = 0.0  # Assuming no gripper action for simplicity

        rospy.Subscriber('/vive/controller/right/pose', PoseStamped, self.pose_callback)

        # Debug: Ensure demo_replay_buffer is not None
        print(f"Initialized XarmEnvCollectData with demo_replay_buffer: {type(self.demo_replay_buffer)}")

    def pose_callback(self, data):
        current_position = np.array([-data.pose.position.x - 0.8, -data.pose.position.y - 0.78, data.pose.position.z - 0.7])
        relative_position = (current_position - self.prev_position)*[110,85,100]
        self.relative_positions.append(relative_position)
        self.prev_position = current_position
        # print("self.relative_position", self.relative_positions.pop(0))

    def collect_human_demonstrations(self):
        
        rate = rospy.Rate(10)  # 10 Hz
        
        for _ in range(self.num_episodes):
            obs = self.env.reset()
            for i in range(50):
                if self.relative_positions:
                    rel_pos = self.relative_positions.pop(0)
                    action = np.append(rel_pos, self.gripper_action)
                    print("action", action)
                    next_obs, reward, done, info = self.env.step(action)
                    env.render()
                    
                    demo_replay_buffer.add(obs, next_obs, action, reward, done, infos=info)
                    print("demoreplaybuffer",demo_replay_buffer.pos)

                    obs = next_obs
                    if done or info.get("is_success", False):
                        print("Success?", info.get("is_success", False))
                        print("Safe?", info.get("is_safe", False))
                        time.sleep(1./240)
                        obs = self.env.reset()
                rate.sleep()

        # self.env.close()


if __name__ == '__main__':

  rospy.init_node('Retrain_xarm_gym_sac', anonymous=True)

  config = {
    'GUI': False, 
    'reward_type':'dense_diff',
    'Sim2Real': False,
    'Digital_twin': True
  }

  #initial positon of eef in pybullet 
  initial_position = [-9.30021275e-02, -6.19335179e-07, 1.11998452e-01] 

  # Set the logging system
  rospack = rospkg.RosPack()
  pkg_path = rospack.get_path('gym_ros_pybullet')
  outdir = pkg_path + '/train' # dir for saving the trained model
  logdir = pkg_path + '/log'  # dir for tensorboard
  os.makedirs(outdir, exist_ok=True)

  # Create the Gym environment
  env = gym.make('Gym_xarm/XarmReachSac-v0', config = config, render_mode="rgb_array")
  env = Monitor(env, outdir)
  rospy.loginfo ( "Monitor Wrapper started")
  rospy.loginfo ( "Gym environment done")
  # time.sleep(15)

  # Initialize the custom replay buffer
  buffer_size = 50
  action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
  observation_space = spaces.Dict({
    "observation": spaces.Box(-np.inf, np.inf, shape=(18,), dtype='float32'),
    "achieved_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
    "desired_goal": spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
    "obstacles": spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
  })
    
  # demo_replay_buffer = DictReplayBuffer(buffer_size, observation_space, action_space, handle_timeout_termination = False) # 60% sampled from human demonstrations
  # print(f"Initialized demo_replay_buffer: {type(demo_replay_buffer)}")

  # # Initialize the controller for collect human demonstraion
  # controller = XarmEnvCollectData(env, initial_position, demo_replay_buffer, num_episodes=1) # num_episodes mean how many trajectories that you get from human
  # # Run the controller

  # print("==============================================================================")
  # print("Please move the VR controller to collect data!")
  # print("==============================================================================")
  # controller.collect_human_demonstrations()

  # save_to_pkl(path=f"{outdir}/human_demo_pkl", obj=demo_replay_buffer)
  # print(f"The loaded_model has {demo_replay_buffer.size()} transitions in its buffer")

  # ######################################################################################################

  retain_timesteps = rospy.get_param("/retain_timesteps")

  demo_replay_buffer = load_from_pkl(path=f"{outdir}/Non-optimal_expert_demo_v3") # Expert_demo_best # Non-optimal_expert_demo_v3
  # print("------------------------Initialising SACBC-------------------------------------")
  # print("Before SACBC, deomoreplybuffer,POS is", demo_replay_buffer.pos)
  # model = SACBC(demo_data=demo_replay_buffer, policy="MultiInputPolicy", env=env, verbose=1)
  # print(f"Initialized SACBC model with demo_data: {type(model.demo_data)}")
  # print("-------------------Finished Initialising SACBC---------------------------------")

  loaded_model = SACBC.load(f"{outdir}/SAC_v1", demo_replay_buffer, env=env, use_sde=True)
  loaded_model.load_replay_buffer(f"{outdir}/SAC_Replaybuffer_v1")
  # loaded_model.demo_data=demo_replay_buffer

  # Create the callback: check every 1000 steps
  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=outdir)
  loaded_model.learn(retain_timesteps, callback=callback, tb_log_name="sacbc_nonoptimal_Qfilter_v15", reset_num_timesteps=False)

  loaded_model.save(f"{outdir}/SAC_retrained_with_demo")
  
  rospy.loginfo("Re-training done!")

  env.close()


