# -*- coding: utf-8 -*-
import numpy as np


def discount_rewards(rewards):
    discounted_rewards = []
    for reward in rewards:
        tmp = (reward - np.mean(reward)) / (np.std(reward)+1e-4)
        discounted_rewards.append(tmp)
    return discounted_rewards
            