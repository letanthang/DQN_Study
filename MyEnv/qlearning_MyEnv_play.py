import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
from qlearning_env_train import Blob

style.use('ggplot')

#env params
SIZE = 10
EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
SHOW_EVERY = 1
ACTION_NUM = 200

# load q_table
start_q_table = 'qtable-1651474074.pickle' # pretrained Q-table

#keys in COLORS dict
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

#in cv2 img is bgr
CYAN = (255,255,100)
GREEN = (0,250,0)
RED = (0,0,250)
COLORS = {1:CYAN, 2:GREEN, 3:RED}


with open(start_q_table,'rb') as f:
		q_table = pickle.load(f)

episode_rewards = []

for episode in range(EPISODES):
	player = Blob()
	food = Blob()
	enemy = Blob()

	if episode % SHOW_EVERY:
		print(f'on episode # {episode}')
		print(f'{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}')
		render = True
	else:
		render = False

	episode_reward = 0

	for i in range(ACTION_NUM):
		obs = (player-food, player-enemy)
		action = np.argmax(q_table[obs])		
		player.action(action)
		enemy.move()
		food.move()

		if player.x == enemy.x and player.y == enemy.y:
			reward = -ENEMY_PENALTY
		elif player.x == food.x and player.y == food.y:
			reward = FOOD_REWARD
		else:
			reward = -MOVE_PENALTY
		
		if render:
			env = np.zeros((SIZE,SIZE,3), dtype=np.uint8)
			env[food.x][food.y] = COLORS[FOOD_N]
			env[player.x][player.y] = COLORS[PLAYER_N]
			env[enemy.x][enemy.y] = COLORS[ENEMY_N]
			img = Image.fromarray(env, 'RGB')
			cv.namedWindow("MyEnv", cv.WINDOW_NORMAL) 
			cv.imshow('MyEnv', np.array(img))
			if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
				if cv.waitKey(500) & 0xFF == ord('q'):
					break
			else:
				if cv.waitKey(1) & 0xFF == ord('q'):
					break

		episode_reward += reward
		if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
			break

	episode_rewards.append(episode_reward)

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'reward {SHOW_EVERY} ma')
plt.xlabel('Episode #')
plt.show()
	