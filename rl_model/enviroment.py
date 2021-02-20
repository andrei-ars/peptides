# based on gridworld.py and tictactoe.py
"""
Requirements
pip install lxml
"""

import datetime
import os
import sys
import json
import logging
from collections import deque
import numpy as np
import torch
#from gym.utils import seeding
#from webdriver import SeleniumWebDriver
#from webdriver_imitation import WebDriverImitaion
from database import DataBase

NUMBER_ACTIONS = 12 # = self.output_size # max number of possible measure for a file
#self.max_total_steps = 20
WIN_REWARD, POS_REWARD, NEG_REWARD = 50, 0, 0  # 100, 1, 0


def negative_reward():
    #print(NEG_REWARD)
    return NEG_REWARD


def positive_reward():
    #print(POS_REWARD)
    return POS_REWARD


class Enviroment:
    """This is enviroment for a test driver
    """
    def __init__(self, max_total_steps=100000000):

        self.env_size = (12, 32) # (3, 5)
        self.max_steps = 6 # the number of files for peptide
        #self.max_total_steps = max_total_steps

        self.step_count = 0
        self.total_steps = 0
        self.last_num_steps = 0
        self.history_steps = deque()
        self.history_rewards = deque()
        self.history_error = deque()
        self.history = []
        #self.seed()
        #self.reset()
        self.wins = 0
        self.losses = 0

        self.possible_actions = list(range(NUMBER_ACTIONS))
        self.database = DataBase(path="data")
        self.peptide_count = 0
        self.action_index =None

        #with open("targets.json") as fp:
        #    self.targets = json.load(fp)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    #def to_play(self):
    #    return 0 if self.player == 1 else 1

    def reset(self):
        #self.board = numpy.zeros((3, 3)).astype(int)
        #self.state = [0]
        self.last_num_steps = self.step_count
        self.step_count = 0
        self.prv_state = None
        #self.driver.reset()
        #data = self.database.get_data_for_peptide(self.peptide_count)
        self.peptide_count += 1
        self.history = []

        return self.get_observation_float()

    def step(self, action):
        """
        action (int or str)
        """
        #log = self.driver.logger.log
        self.step_count += 1
        self.total_steps += 1

        # Call the webdriver and perform the action
        if type(action) is str:
            cmd = action
        else:
            action = int(action)

        #if len(self.history) > 0 and self.history[-1].get('cmd') == "ENTER" and cmd == "ENTER":
        #    enter_repeating = True
        #else:
        #    enter_repeating = False
        #self.history.append({'cmd':cmd})

        #reward = 0
        #site_elements = self.driver.get_site_elements()
        #current_element = None

        #reward = 1 if self.have_winner() else 0
        #if self.have_winner():
        #    reward = self.get_final_reward()
        #    self.wins += 1
        #    self.history_steps.append(self.step_count)
        #    if len(self.history_steps) > 100:
        #        self.history_steps.popleft()
        #    avg_steps = np.mean(self.history_steps)
        #    
        #    log("site_elements (final): {}".format(site_elements))
        #    log("{}-th win in {} steps [{:.1f}]; reward={:.4f}".format(
        #        self.wins, self.step_count, avg_steps, reward))
        #    self.last_num_steps = self.step_count
        #    self.step_count = 0

        index = self.action_index[action]
        meta_data = self.database.get_meta_data(index)
        self.history.append({
                'count': self.step_count,
                'action': action,
                'sample_type': meta_data['sample_type'],
                'ClusterMean': meta_data['ClusterMean'],
                'value': meta_data['value']
                })

        if self.step_count == self.max_steps - 1:
            reward = self.calculate_reward(self.history)
            done = True
        else:
            reward = 0
            done = False

        #print("step={}: reward={}".format(self.step_count, reward))

        if done:
            #print("Done: {}. Reset.".format(done))
            self.reset()

        #if self.step_count == self.max_total_steps + 1:
        #    self.driver.print_state()

        return self.get_observation_float(), reward, done, {}

    def calculate_reward(self, history):
        a_values = []
        b_values = []
        for sample in history:
            if sample['sample_type'] == 'A':
                a_values.append(sample['ClusterMean'])
            elif sample['sample_type'] == 'B':
                b_values.append(sample['ClusterMean'])
            else:
                raise Exception("wrong sample_type")

        target_value = history[0]['value']
        a_mean = np.mean(a_values)
        b_mean = np.mean(b_values)
        value = a_mean - b_mean
        error = float(abs(target_value - value))
        reward = 1 - error

        #self.history_rewards.append(reward)
        self.history_error.append(error)
        if len(self.history_error) > 100:
            self.history_error.popleft()
        avg_error = np.mean(self.history_error)
        print("value: {:.4f} [target={:.4f}], error: {:.4f} [avg={:.4f}]".format(value, target_value, error, avg_error))

        return reward

    #def get_final_reward(self):
    #    return WIN_REWARD * (1 - 0.8*(self.step_count / self.max_total_steps))

    #def get_site_elements(self):
    #    """ For debuging
    #    """
    #    site_elements = self.driver.get_site_elements()
    #    print("site_elements:", site_elements)
    #    return site_elements

    def get_observation(self):
        # It should return the current state as a numpy array of the float32 type
        # i.e. whole necessary information from the current webpage
        # including the list of active elements and so on.
        # probably, some additional information.
        # It will be fed to neural network.

        file = self.database.get_file_info(peptide_number=self.peptide_count, file_number=self.step_count)
        # shape (12, 32)
        
        state = file['array']
        self.action_index = file['indices']

        #(hight, width) = self.env_size
        #env_state = [[1 if i<lengths[j] and site_elements[de_type[j]][i] else 0 for i in range(width)] for j in range(len(lengths))]
        #env_state = np.array(env_state, dtype=np.int32)
        #int_state = np.zeros((hight, width), dtype=np.int32)
        #de_type_to_number = {y:x for x,y in de_type.items()}

        #if self.element_type is not None and self.element_number is not None:
        #    de_type_number = de_type_to_number.get(self.element_type)
        #    int_state[de_type_number, self.element_number] = 1

        #if self.prv_state is None:
        #    self.prv_state = env_state

        #state = np.vstack([env_state, self.prv_state, int_state])
        #state = np.expand_dims(state, axis=0) # 3-dim. array is required
        #state = state.flatten()
        #self.prv_state = env_state

        return state

    def get_observation_float(self):
        #return np.array(self.get_observation(), dtype=np.float32)
        return np.array(self.get_observation(), dtype=np.float32)

    def legal_actions(self):
        legal = list(range(len(self.possible_actions)))
        return legal

    def have_winner(self):

        # Usage of observation
        observation = self.get_observation()
        #html = self.get_html()
        #sum_obs = np.sum(observation)  # whole state (including inner state)
        #sum_obs = np.sum(observation[:,:3])  # only env_state
        sum_obs = np.sum(observation[:self.env_size[0]*self.env_size[1]])
        #print("sum_obs={}, wins={}\n{}".format(sum_obs, self.wins, str(observation)))
        #print("sum_obs={}, wins={}".format(sum_obs, self.wins))
        # if all_active_elements_have_been_clicked
        is_zero_observation = True if sum_obs < 0.01 else False

        # Checking whether the page has been updated
        is_target_achieved = self.driver.is_target_achieved(self.targets)

        #return is_zero_observation
        return is_target_achieved

    def render(self):
        print("Display the game observation")
        print(self.get_observation())

    def obs_size(self):
        #return 3 * self.env_size[0] * self.env_size[1]
        return 12 * 32

    def number_of_actions(self):
        return NUMBER_ACTIONS


if __name__ == "__main__":

    env = Enviroment(max_total_steps=100)
    print(env.get_observation())
    print(env.obs_size())
    print(env.legal_actions())
    env.step(0)
    env.step(2)
    env.step(5)
    env.step(1)
    env.step(3)
    env.step(4)
    env.step(0)
    env.step(1)
    #env.step("CHOOSE_FIRST_CLICK")
    #env.step("NEXT")
    #env.step("CLICK")
    #print(env.get_observation())
    #elements = env.get_site_elements()
    #print(elements)
    #env.driver.reset()
    #env.driver.driver.get(init_url_address)

    #print("\n\nTest game")
    #game = Game()
    #game.step("WAIT")
    #game.step("CHOOSE_FIRST_CLICK")
    #game.step("NEXT")
    #game.step("CLICK")
