from curses import COLORS
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

style.use('ggplot')

#env params
SIZE = 10
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25

#agent learning settings
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000
start_q_table = None
LEARNING_RATE = 0.1
DISCOUNT = 0.95

#
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

#in cv2 img is bgr
CYAN = (255,255,100)
GREEN = (0,250,0)
RED = (0,0,250)

COLORS = {1:CYAN, 2:GREEN, 3:RED}

