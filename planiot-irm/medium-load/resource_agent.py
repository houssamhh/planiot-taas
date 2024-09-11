# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 13:47:47 2024

@author: houss
"""
import pandas as pd

# Import Gym
from gymnasium import Env
from gymnasium.spaces import Discrete, MultiDiscrete

# Import helpers
import numpy as np
import random
import os
import subprocess
import csv


# Import Stable baselines
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

app_categories_file = './experiments/medium-load/app_categories.csv'

resource_allocation_map = {0: 1, 1: 1.2, 2: 1.4, 3: 1.5, 4: 1.7}
bandwidth_map = {0: 650, 1: 670, 2: 700, 3: 750}

output_file = './experiments/100subs/results/output.csv'


def getAppCategories(file):
    app_categories = dict()
    df = pd.read_csv(file)
    for index, row in df.iterrows():
        app = row['app']
        category = row['category']
        app_categories[app] = category
    return app_categories
    
# returns response time per app category (dict)
# def getResponseTimePerCategory(output_file, config):
    # app_categories = getAppCategories(app_categories_file)
    # responsetimes = {'AN': 0, 'RT': 0, 'TS': 0, 'VS': 0}
    # nb_subscriptions = {'AN': 0, 'RT': 0, 'TS': 0, 'VS': 0}
    # df = pd.read_csv(output_file, usecols=['app', 'topic', config+'.json'])
    # for index, row in df.iterrows():
    #     app = row['app']
    #     response_time = row[config+'.json']
    #     app_category = app_categories[app]
    #     responsetime_sum = responsetimes[app_category]
    #     responsetime_sum += response_time
    #     responsetimes[app_category] = responsetime_sum
    #     subscriptions = nb_subscriptions[app_category]
    #     subscriptions += 1
    #     nb_subscriptions[app_category] = subscriptions
    # for category in nb_subscriptions.keys():
    #     responsetimes[category] = responsetimes[category] / nb_subscriptions[category]
    # return responsetimes
        
    
def get_app_categories():
    appCategories = dict()
    with open (app_categories_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            app = row['app']
            category = row['category']
            appCategories[app] = category       
    return appCategories

def get_response_times_per_subscription(fileName):
    responseTimes = dict()
    categories = get_app_categories()
    with open (fileName, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            topic = row['topic']
            app = row['app']
            responsetime = row['response_time']
            cat = categories[app]
            responseTimes[topic + '_' + app + '_' + cat] = float(responsetime)
        return responseTimes
    
def get_response_times_per_category(fileName):
    responseTimes = get_response_times_per_subscription(fileName)
    categories = get_app_categories()
    nbAN, nbRT, nbTS, nbVS, nbEM = 0, 0, 0, 0, 0
    for app in responseTimes.keys():
        cat = categories[app.split('_')[1]]
        if cat == 'AN':
            nbAN += 1
        elif cat == 'RT':
            nbRT += 1
        elif cat == 'TS':
            nbTS += 1
        elif cat == 'VS':
            nbVS += 1
        elif cat == 'EM':
            nbEM += 1
    responsetimeAN, responsetimeRT, responsetimeTS, responsetimeVS, responsetimeEM = 0, 0, 0, 0, 0
    responsetimeAN, responsetimeRT, responsetimeTS, responsetimeVS = 0, 0, 0, 0
    for subscription, value in responseTimes.items():
        app = subscription.split('_')[1]
        if categories[app] == 'AN':
            responsetimeAN += value
        elif categories[app] == 'RT':
            responsetimeRT += value
        elif categories[app] == 'TS':
            responsetimeTS += value
        elif categories[app] == 'VS':
            responsetimeVS += value
        elif categories[app] == 'EM':
            responsetimeEM += value
    responsetimeAN = responsetimeAN / nbAN
    responsetimeRT = responsetimeRT / nbRT
    responsetimeTS = responsetimeTS / nbTS
    responsetimeVS = responsetimeVS / nbVS
    response_times_dict = dict()
    response_times_dict['AN'] = responsetimeAN
    response_times_dict['RT'] = responsetimeRT
    response_times_dict['TS'] = responsetimeTS
    response_times_dict['VS'] = responsetimeVS
    return response_times_dict

    
def calculateReward(responsetimes):
    reward = 0
    reward = responsetimes['RT'] + responsetimes['VS'] + responsetimes['TS'] + responsetimes['AN']
    reward = reward * -1
    return reward

# config = '0.9x'
# print(getResponseTimePerCategory(output_file, config))


class IotEnv(Env):
    config = 'default'
    def __init__(self):
        
        # Resource allocation possibilities: (resources, bandwidth)
        self.action_space = Discrete(8)        
        
        # current resource allocation
        self.observation_space = Discrete(8)
        
        # 
        self.state = Discrete(8).sample()
        
        # we just want to take one action in an episode
        self.config_length = 1
        
    def step(self, action):
        global index
        config = 'default'
        # resource_allocation = resource_allocation_map[action[0]]
        bandwidth_allocation = (action + 1) * 30 + 230
        print ("\nAction: {}".format(bandwidth_allocation))

        metrics_file = 'experiments/medium-load/jsimg/default-{}.jsimg-result.jsim'.format(bandwidth_allocation)
        responsetimes = get_response_times_per_category(metrics_file)
        print(responsetimes)
        reward = calculateReward(responsetimes)
        self.config_length -= 1
        
        
        print('\nReward: {}'.format(reward), '\n')
        print("********************************\n\n")
        
        
        if self.config_length <= 0:
            done = True
        else:
            done = False
            
        info = {}
        truncated = False
        
        
        return self.state, reward, done, truncated, info
    
    def render(self):
        # Implement visualization if needed
        pass
    
    def reset(self, seed=None, options=None):
        bandwidth_allocation = random.randint(0, 7)
        self.state = bandwidth_allocation
        self.config_length = 1
        return self.state, {}
    
    
env = IotEnv()
env.reset()

model_name = 'ppo_medium_700ts'
save_path = os.path.join('experiments', 'training', 'iot', 'models', model_name)
# model = PPO('MlpPolicy', env, verbose=2)
# print('\n\n\n**********Training Model*************\n\n\n')
# model.learn(total_timesteps=700)
# print('\n\n\n**********Saving Model*************\n\n\n')
# model.save(save_path)

model = PPO.load(save_path, env=env)
print(model.predict((0), deterministic=True))
print(evaluate_policy(model, env, n_eval_episodes=10))
    