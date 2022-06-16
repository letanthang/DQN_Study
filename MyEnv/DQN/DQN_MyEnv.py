from collections import deque
import torch
import numpy as np
import random
from tqdm import tqdm
from Game_Environment import BlobEnv
from DQN_model import DQNmodel, QTrainer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

REPLAY_MEMORY_SIZE = 50_000     # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64             # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5         # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200               # For model save

MAX_MEMORY = 100_000
LR = 0.001
BATCH_SIZE = 1000

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


env = BlobEnv()
# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
torch.random.manual_seed(1)

class DQNAgent:
    input_dim = env.OBSERVATION_SPACE_VALUES[2]
    output_dim = env.ACTION_SPACE_SIZE
    def __init__(self):
        # main model    #get trained every step
        self.model = DQNmodel(self.input_dim, self.output_dim).to(device)

        # Targer model - this is what we predict against every step
        self.target_model = DQNmodel(self.input_dim, self.output_dim).to(device)
        self.target_model.load_state_dict(self.model.state_dict())

        # trainer
        self.trainer = QTrainer(self.model, lr=LR, gamma=DISCOUNT)

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
        current_state = torch.Tensor(np.array(state)).reshape(-1, env.SIZE, env.SIZE)/255.0
        return self.model(current_state)[0]

    def learn(self, terminal_state, step):
        # Start training only if certain number of samples is already saved
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return
        
        # Get a minibatch of random samples from memory replay table
        minibatch =  random.sample(self.replay_memory, MINIBATCH_SIZE)
        current_state, action, reward, new_current_state, done = zip(*minibatch)
        current_states = np.array([transition[0].reshape(-1, env.SIZE, env.SIZE) for transition in minibatch])/255.0
        new_current_states = np.array([transition[3].reshape(-1, env.SIZE, env.SIZE) for transition in minibatch])/255.0
        self.trainer.train_step(current_states, action, reward, new_current_states, done) 

        
        # updating to det if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0


agent = DQNAgent()

for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):
    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()

    # Reset flag and start iterating until episode ends
    done = False
    while not done:
        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = torch.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.learn(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)

    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
