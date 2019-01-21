import matplotlib
matplotlib.use("TkAgg")

import copy
import pylab
import random
import numpy as np

from environment import Env # Show canvas
# from environment2 import Env # For Train

from keras.layers import Dense, Conv1D, Flatten
from keras.optimizers import Adam
from kerasa.models import Sequential
from keras import backend as K

EPISODES = 3000

class PolicyGradient_Agent:
    def __init__(self, state_size):
        self.load_model = False
        if self.load_model:
            self.model.load_weights("./save_model/PolicyGradient_Agent.h5")
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.state_size = state_size
        self.discount_factor = 0.99
        self.learning_rate = 0.0001

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters = 10, kernel_size = 1, input_shape = (self.state_size, 3), kernel_initializer = 'he_uniform', activation = 'tanh'))
        model.add(Dense(30, kernel_initializer = 'he_uniform', activation = 'tanh'))
        model.add(Dense(30, kernel_initializer = 'he_uniform', activation = 'tanh'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation = 'softmax')) # Output : Policy
        model.summary()
        return model

    def optimizer(self):
        action = K.placeholder(shape = [None, self.action_size])
        discounted_rewards = K.placeholder(shape = [None, ])

        # Cross Entropy
        action_prob = K.sum(action * self.model.output, axis = 1)
        cross_entropy = K.log(action_prob) * discounted_rewards
        loss = -K.sum(cross_entropy)

        # Optimizer with Train
        optimizer = Adam(lr = self.learning_rate)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, action, discounted_rewards], [], updates = updates)
        return train
    
    def get_action(self, state):
        policy = self.model.predict(state)[0]
        return np.random.choice(self,action_size, 1, p = policy)[0]
    
    # Calculate Return Value
    def discount_rewards(self, rewards):
        discounted_rewards = np.zero_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # Update Replay memory <state, action, reward>
    def append_sample(self, state, action, reward):
        self.stattes.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)
    
    # Model Training
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []

if __name__ == "__main__":
    env = Env()
    state = env.reset()
    state_size = env.state_size

    # Agent
    agent = PolicyGradient_Agent(state_size)
    globaal_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        score, step, done = 0, 0, False
        
        state = env.reset()
        state = np.expand_dims(state, axis = 0)

        while not done:
            global_step += 1
            step += 1
            action = agent.get_action(state)
            if global_step == 1:
                policy = [0.25, 0.25, 0.25, 0.25]
            if step > 800:
                action = random.randint(0, 3)
            
            next_state, reward, done, goal = env.step(action)

            if step == 1000:
                reward -= 100
                done = True
            
            if goal == True:
                reward += 100
            
            next_state = np.expand_dim(next_state, axis = 0)
            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next)state)

            if done:
                if goal == 1:
                    goal = "Goal!"
                else:
                    goal = "No Goal..."
                
                agent.train_model()
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/PolicyGradient_Agent.png")
                print("episode: ", e, "score: ", score, "global_step: ", global_step, "goal: ", goal)

        # Save weight
        if e % 10 == 0:
            agent.model.save_weights("./save_model/PolicyGradient_Agent.h5")
    print('Mean Step : ', (global_step / EPISODES), 'Survival rate : ', (1 - total_dead / (3 * EPISODES)),
          'Collection rate : ', (1 - total_remain_oil / (3 * EPISODES)))