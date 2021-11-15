# import gym
import os
# import mujoco_py
from agent import Agent
from train import Train
# from play import Play

# ENV_NAME = "Swimmer"
# TRAIN_FLAG = False
# test_env = gym.make(ENV_NAME + "-v2")

import logging
import math
import time
import sys
import os
import copy
from matplotlib import pyplot as plt 

import numpy as np
import torch
import torch.autograd
import ipdb
from envs import PendulumDynamics, AcrobotEnv

# ENV_NAME = "PendulumCont-v0"
ENV_NAME = "AcrobotCont-v0"


# n_states = test_env.observation_space.shape[0]
# action_bounds = [test_env.action_space.low[0], test_env.action_space.high[0]]
# n_actions = test_env.action_space.shape[0]

n_iterations = 600
lr = 3e-4
epochs = 10
clip_range = 0.2
mini_batch_size = 64
T = 250 
N_BATCH = 100
LQR_ITER = 60
ACTION_LOW = -2.0
ACTION_HIGH = 2.0
batch_size = 64
device = torch.device('cpu')
mini_batch_size = 512
n_states = 4
action_bounds = [-2, 2]
n_actions = 1
TRAIN_FLAG=True
# test_env = PendulumDynamics(batch_size=batch_size,device=device)
# env = PendulumDynamics(batch_size=batch_size,device=device)

test_env = AcrobotEnv(batch_size=batch_size,device=device)
env = AcrobotEnv(batch_size=batch_size,device=device)

if __name__ == "__main__":
    print(f"number of states:{n_states}\n"
          f"action bounds:{action_bounds}\n"
          f"number of actions:{n_actions}")

    if not os.path.exists(ENV_NAME):
        os.mkdir(ENV_NAME)
        os.mkdir(ENV_NAME + "/logs")

    # env = gym.make(ENV_NAME + "-v2")

    agent = Agent(n_states=n_states,
                  n_iter=n_iterations,
                  env_name=ENV_NAME,
                  action_bounds=action_bounds,
                  n_actions=n_actions,
                  lr=lr,
                  device=device)
    if TRAIN_FLAG:
        trainer = Train(env=env,
                        test_env=test_env,
                        env_name=ENV_NAME,
                        agent=agent,
                        horizon=T,
                        n_iterations=n_iterations,
                        epochs=epochs,
                        mini_batch_size=mini_batch_size,
                        epsilon=clip_range)
        trainer.step()

    player = Play(env, agent, ENV_NAME)
    player.evaluate()
