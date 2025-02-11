import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from imitation.algorithms import bc
# import our custom training environment
import Gym_xarm


env = gym.make("CartPole-v1")
env = Monitor(env)
expert = PPO(
    policy=MlpPolicy,
    env=env,
    seed=0,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0003,
    n_epochs=10,
    n_steps=64,
)
expert.learn(3000)  # Note: set to 100000 to train a proficient expert
reward, _ = evaluate_policy(expert, env, 10)
print('Pre-trained expert reward:',reward)

rng = np.random.default_rng()
rollouts = rollout.rollout(
    expert,
    DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
    rollout.make_sample_until(min_timesteps=None, min_episodes=2),
    rng=rng,
)
transitions = rollout.flatten_trajectories(rollouts)
print('Sample the datasets from expert:')
print(
    f"""The `rollout` function generated a list of {len(rollouts)} {type(rollouts[0])}.
After flattening, this list is turned into a {type(transitions)} object containing {len(transitions)} transitions.
The transitions object contains arrays for: {', '.join(transitions.__dict__.keys())}."
"""
)


bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=transitions,
)

reward_before_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward before BC training: {reward_before_training}")

bc_trainer.train(n_epochs=20)
reward_after_training, _ = evaluate_policy(bc_trainer.policy, env, 10)
print(f"Reward after BC training: {reward_after_training}")