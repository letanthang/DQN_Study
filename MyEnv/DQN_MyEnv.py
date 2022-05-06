from collections import deque
from re import X
import time
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
import os
from PIL import Image
import cv2 as cv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

REPLAY_MEMORY_SIZE = 50_000     # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64             # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5         # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200               # For model save
# MEMORY_FRACTION = 0.20

# env settings
DISCOUNT = 0.99
EPISODES = 20_000

# Exploration settings
epsilon = 1
EPSILON_DECAY = 0.9998
MIN_EPSILON = 0.001

#stats setting
AGGREGATE_STATS_EVERY = 50 
SHOW_PREVIEW = False

class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

        elif choice == 4:
            self.move(x=1, y=0)
        elif choice == 5:
            self.move(x=-1, y=0)

        elif choice == 6:
            self.move(x=0, y=1)
        elif choice == 7:
            self.move(x=0, y=-1)

        elif choice == 8:
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1

class BlobEnv:
    SIZE = 10
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)  # 4
    ACTION_SPACE_SIZE = 9
    PLAYER_N = 1  # player key in dict
    FOOD_N = 2  # food key in dict
    ENEMY_N = 3  # enemy key in dict
    # the dict! (colors)
    d = {1: (255, 175, 100),
         2: (0, 255, 0),
         3: (0, 0, 255)}

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        while self.food == self.player:
            self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        # self.enemy.move()
        # self.food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        done = False
        if reward == self.FOOD_REWARD or reward == -self.ENEMY_PENALTY or self.episode_step >= 200:
            done = True

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        cv.namedWindow("MyEnv", cv.WINDOW_NORMAL) 
        cv.imshow('MyEnv', np.array(img))
        cv.waitKey(1)

    # FOR CNN #
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)   # starts an rbg of our size
        env[self.food.x][self.food.y] = self.d[self.FOOD_N]         # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.d[self.ENEMY_N]      # sets the enemy location to red
        env[self.player.x][self.player.y] = self.d[self.PLAYER_N]   # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')                           # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img

env = BlobEnv()

# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
torch.random.manual_seed(1)

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# class DQNmodel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         input_dim = env.OBSERVATION_SPACE_VALUES
#         output_dim = env.ACTION_SPACE_SIZE
#         c, h, w = input_dim
#         self.online = nn.Sequential(
#             nn.Conv2d(in_channels=c, out_channels=256, kernel_size=3),          #env obs channels = 3
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             # nn.Conv2d(256, 256, kernel_size=3),
#             # nn.ReLU(),
#             # nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Dropout(0.2),
#             nn.Flatten(),
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, output_dim)                                     #env action channels = 9
#         )

#     def forward(self, x):
#         x = self.online(x)
#         output = F.log_softmax(x, dim=1)
#         return output

class DQNmodel(nn.Module):
    def __init__(self):
        super().__init__()
        input_dim = env.OBSERVATION_SPACE_VALUES
        output_dim = env.ACTION_SPACE_SIZE
        c, h, w = input_dim
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=256, kernel_size=3)          #env obs channels = 3
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, output_dim)                                     #env action channels = 9


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class DQNAgent:
    def __init__(self):
        # main model    #get trained every step
        self.model = DQNmodel().to(device)

        # Targer model - this is what we predict against every step
        self.target_model = DQNmodel().to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        # An array with last n steps for training
        self.replay_memory = deque(maxlen = REPLAY_MEMORY_SIZE)

        # Used to count when to update target network with main network's weights   
        self.target_update_counter = 0

    # Adds step's data to a memory replay array
    # (observation space, action, reward, new observation space, done)
    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

     # Queries main network for Q values given current observation space (environment state)
    def get_qs(self, state):
        current_state = torch.Tensor(np.array(state)).reshape(-1, *state.shape)/255
        self.model.eval()
        return self.model(current_state[0])

    def learn(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get a minibatch of random samples from memory replay table
        minibatch =  random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([transition[0] for transition in minibatch])/255
        current_states = torch.Tensor(current_states).reshape(-1,env.SIZE,env.SIZE)
        self.model.eval()
        current_qs_list = self.model(current_states)[0]

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        new_current_states = np.array([transition[3] for transition in minibatch])/255
        new_current_states = torch.Tensor([i[0] for i in new_current_states]).reshape(-1,env.SIZE,env.SIZE)
        self.target_model.eval()
        future_qs_list = self.target_model(new_current_states)[0]

        X = []
        y = []

        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            # If not a terminal state, get new q from future states, otherwise set it to 0
            # almost like with Q Learning, but we use just part of equation here
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

             # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)
        
        X = torch.Tensor([i[0] for i in X]).reshape(-1,env.SIZE,env.SIZE)
        X = X/255.0
        y = torch.Tensor([i[1] for i in y])

        # Fit on all samples as one batch, log only on terminal state
        for i in tqdm(range(0, len(X), MINIBATCH_SIZE)):
            batch_X = X[i:i+MINIBATCH_SIZE].reshape(-1,3,env.SIZE,env.SIZE)
            batch_y = y[i:i+MINIBATCH_SIZE]
            self.model.train()
            self.model.zero_grad()
            outputs = self.model(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            opt.step()

        # self.model.fit (np.array(X)/255, np.array(y), batch_size=MINIBATCH_SIZE, verbose=0, shuffle=False)

        #updating to det if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0


agent = DQNAgent()
opt = optim.Adam(agent.model.parameters(), lr=0.001)
loss_function = nn.MSELoss()
current_state = env.reset()
agent.model.eval()
print(agent.model(torch.tensor(current_state.reshape(-1, *current_state.shape)/255.0)))
print(current_state.shape)
# env.render()
# cv.waitKey()

# for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
#     # Restarting episode - reset episode reward and step number
#     episode_reward = 0
#     step = 1

#     # Reset environment and get initial state
#     current_state = env.reset()

#     # Reset flag and start iterating until episode ends
#     done = False
#     while not done:
#         # This part stays mostly the same, the change is to query a model for Q values
#         if np.random.random() > epsilon:
#             # Get action from Q table
#             action = np.argmax(agent.get_qs(current_state))
#         else:
#             # Get random action
#             action = np.random.randint(0, env.ACTION_SPACE_SIZE)

#         new_state, reward, done = env.step(action)

#         # Transform new continous state to new discrete state and count reward
#         episode_reward += reward

#         if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
#             env.render()

#         # Every step we update replay memory and train main network
#         agent.update_replay_memory((current_state, action, reward, new_state, done))
#         agent.learn(done, step)

#         current_state = new_state
#         step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    # ep_rewards.append(episode_reward)
    # if not episode % AGGREGATE_STATS_EVERY or episode == 1:
    #     average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
    #     min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
    #     max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

    #     # Save model, but only when min reward is greater or equal a set value
    #     if min_reward >= MIN_REWARD:
    #         torch.save(agent.model, f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # # Decay epsilon
    # if epsilon > MIN_EPSILON:
    #     epsilon *= EPSILON_DECAY
    #     epsilon = max(MIN_EPSILON, epsilon)