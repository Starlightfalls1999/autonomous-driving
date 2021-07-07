import argparse
import gym
import gym_routing_contiuous_2
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tempfile
import time
import random
import _thread
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

from stable_baselines.deepq.policies import MlpPolicy
from dqn import DQN



zzz_env = gym.make('zzz-v2')


try:
    model = DQN.load("deepq_0529",env=zzz_env)
    print("load saved model")
except:
    model = DQN(MlpPolicy, env=zzz_env) #, verbose=1
    print("build new model")

model.learn(total_timesteps=100000)
model.save("deepq_0529")

del model # remove to demonstrate saving and loading


print("load model to test")
model = DQN.load("deepq_0529")

obs = zzz_env.reset()


while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = zzz_env.step(action)
    #zzz_env.render()
