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
EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
SHOW_EVERY = 2000

#agent learning settings
epsilon = 0.5
EPS_DECAY = 0.9998
start_q_table = None # pretrained Q-table
LEARNING_RATE = 0.1
DISCOUNT = 0.95
ACTION_NUM = 200


#keys in COLORS dict
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

#in cv2 img is bgr
CYAN = (255,255,100)
GREEN = (0,250,0)
RED = (0,0,250)

COLORS = {1:CYAN, 2:GREEN, 3:RED}

class Blob:
	def __init__(self):
		self.x = np.random.randint(0,SIZE)
		self.y = np.random.randint(0,SIZE)

	def __str__(self):
		return f'{self.x}, {self.y}'

	def __sub__(self, other):
		return (self.x - other.x, self.y - other.y)

	def action(self, choice):
		#just move diagonally
		if choice == 0:
			self.move(x=1,y=1)
		if choice == 1:
			self.move(x=-1,y=-1)
		if choice == 2:
			self.move(x=-1,y=1)
		if choice == 3:
			self.move(x=1,y=-1)

	def move(self, x=False, y =False):
		if not x:
			self.x += np.random.randint(-1, 2)
		else:
			self.x += x
		if not y:
			self.y += np.random.randint(-1, 2)
		else:
			self.y += y

		if self.x < 0:
			self.x = 0
		elif self.x > SIZE - 1:
			self.x = SIZE -1
		if self.y < 0:
			self.y = 0
		elif self.y > SIZE - 1:
			self.y = SIZE -1
		
if start_q_table is None:
	q_table = {}
	# observation space (x1, x2), (x2, y2)
	for x1 in range(-SIZE+1, SIZE):
		for y1 in range(-SIZE+1, SIZE):
			for x2 in range(-SIZE+1, SIZE):
				for y2 in range(-SIZE+1, SIZE):
					q_table[((x1,y1),(x2,y2))] =[np.random.uniform(-5, 0) for i in range(4)]

else: 
	#load pretrained q-table
	with open(start_q_table,'rb') as f:
		q_table = pickle.load(f)

episode_rewards = []

for episode in range(EPISODES):
	player = Blob()
	food = Blob()
	enemy = Blob()

	if episode % SHOW_EVERY == 0:
		print(f'on episode # {episode}, epsilon: {epsilon}')
		print(f'{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}')
		render = True
	else:
		render = False

	episode_reward = 0

	for i in range(ACTION_NUM):
		obs = (player-food, player-enemy)
		if np.random.random() > epsilon:
			action = np.argmax(q_table[obs])
		else:
			action = np.random.randint(0, 4)
		
		player.action(action)

		#### maybe later
		# enemy.move()
		# food.move()
		##############

		if player.x == enemy.x and player.y == enemy.y:
			reward = -ENEMY_PENALTY
		elif player.x == food.x and player.y == food.y:
			reward = FOOD_REWARD
		else:
			reward = -MOVE_PENALTY
		
		new_obs = (player-food, player-enemy)
		max_future_q = np.max(q_table[new_obs])
		current_q = q_table[obs][action]

		if reward == FOOD_REWARD:
			new_q = FOOD_REWARD
		elif reward == -ENEMY_PENALTY:
			new_q = -ENEMY_PENALTY
		else:
			new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE *(reward + DISCOUNT * max_future_q)
		
		q_table[obs][action] = new_q
	
		if render:
			env = np.zeros((SIZE,SIZE,3), dtype=np.uint8)
			env[food.x][food.y] = COLORS[FOOD_N]
			env[player.x][player.y] = COLORS[PLAYER_N]
			env[enemy.x][enemy.y] = COLORS[ENEMY_N]

			img = Image.fromarray(env, 'RGB')
			# img = img.resize((300,300))
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
	epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,)) / SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f'reward {SHOW_EVERY} ma')
plt.xlabel('Episode #')
plt.show()

with open(f'qtable-{int(time.time())}.pickle','wb') as f:
	pickle.dump(q_table, f)