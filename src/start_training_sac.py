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
from stable_baselines3 import PPO, SAC, SACBC
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback

# ROS packages required
import rospy
import rospkg

# import our custom training environment
import Gym_xarm

config = {
    'GUI': True, 
    'reward_type':'dense_diff',
    'Sim2Real': False,
    'Digital_twin': False
}

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
        self.save_path = os.path.join(log_dir, "SAC_v2")
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



if __name__ == '__main__':

    rospy.init_node('train_xarm_gym_sac', anonymous=True)

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('gym_ros_pybullet')
    outdir = pkg_path + '/train' # dir for saving the trained model
    logdir = pkg_path + '/log'  # dir for tensorboard
    os.makedirs(outdir, exist_ok=True)

    # Create the Gym environment
    env = gym.make('Gym_xarm/XarmReachSac-v0', config = config, render_mode="rgb_array")
    env = Monitor(env, outdir)
    # env = VecNormalize(env, norm_obs=True, norm_reward=True)
    rospy.loginfo ( "Gym environment done")

    # env = wrappers.Monitor(env, outdir, force=True) 
    # rospy.loginfo ( "Monitor Wrapper started")

    """
    Loads parameters from the ROS param server
    Parameters are stored in a yaml file inside the config directory
    They are loaded at runtime by the launch file
    """
    total_timesteps = rospy.get_param("/total_timesteps")
    # total_timesteps = 60_000
    # Define the parameters for HER
    # goal_selection_strategy = 'future'  # Strategy for selecting goals
    # n_sampled_goal = 4  # Number of HER samples to generate for each actual transition

    # model = SACBC("MultiInputPolicy", 
    #             env, 
    #             verbose=1, 
    #             replay_buffer_class=HerReplayBuffer,
    #             replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,copy_info_dict=False),
    #             tensorboard_log=logdir,
    #             )
    model = SAC("MultiInputPolicy", 
                env=env, 
                verbose=1, 
                replay_buffer_class=DictReplayBuffer,
                use_sde=True,
                tensorboard_log=logdir,
                )

    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=outdir)

    model.learn(total_timesteps, callback=callback, tb_log_name="SAC_firstrun_v3")


    # plt.show()
    # plt.close()

    model.save(f"{outdir}/SAC_XarmReach_v2")
    model.save_replay_buffer(f"{outdir}/SAC_Replaybuffer_v2")

    rospy.loginfo("Training done!")

    # plot_results([outdir], total_timesteps, results_plotter.X_TIMESTEPS, "Training XarmReachEnv with SAC")
    # plt.savefig(f"{outdir}/reward_train_sac_v1.png", bbox_inches='tight')

    rospy.loginfo("Save all the results!")

    env.close()



