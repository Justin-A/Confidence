# 400

import matplotlib

matplotlib.use('TkAgg')

import copy
import pylab
import random
import numpy as np
from collections import deque
from environment2 import Env
from keras.layers import Dense, Conv1D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential
import pandas as pd

EPISODES = 3000


class DQN_Agent:
    def __init__(self, state_size):
        self.load_model = True
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.state_size = state_size

        # Hyper_Parameters
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.epsilon = 1
        self.epsilon_decay = .99999
        self.epsilon_min = 0.1
        self.batch_size = 32
        self.train_start = 50000

        # Model & Target Model
        self.model = self.build_model()
        self.target_model = self.build_model()

        self.memory = deque(maxlen = 100000)

        if self.load_model:
            self.epsilon = 0.1
            self.model.load_weights('./save_model/DQN_Agent.h5')

    # Network
    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters = 10, kernel_size = self.state_size, input_shape = (self.state_size, 3),
                         kernel_initializer = 'he_uniform', activation = 'tanh'))
        model.add(Dense(30, activation = 'tanh', kernel_initializer = 'he_uniform'))
        model.add(Dense(30, activation = 'tanh', kernel_initializer = 'he_uniform'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation = 'linear'))
        model.summary()
        model.compile(loss = 'mse', optimizer = Adam(lr = self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Choose Action by e-greedy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # Random Action
            return random.randrange(self.action_size)
        else:
            # Action by model
            # state = np.float32(state)
            q_values = self.model.predict(state)
            print("Q_Values: ", q_values)
            return np.argmax(q_values[0])

    # <state, action, reward, next_state> in Replay Memory
    def append_sample(self, state, action, reward, next_state, done):
        # state = np.reshape(state, [1, -1])
        # next_state = np.reshape(next_state, [1, -1])
        self.memory.append((state, action, reward, next_state, done))

    # Random Sampling in Replay Memory, Model Training
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Random Sampling in memory
        mini_batch = random.sample(self.memory, self.batch_size)
        #states = [[0 for x in range(self.state_size)] for y in range(self.batch_size)]
        states = np.zeros((self.batch_size,1 ,self.state_size,3))
        #next_states = [[0 for x in range(self.state_size)] for y in range(self.batch_size)]
        next_states = np.zeros((self.batch_size, 1, self.state_size,3))
        actions, rewards, dones = [], [], []




        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            next_states[i] = mini_batch[i][3]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            dones.append(mini_batch[i][4])

        # states = np.reshape(states, [self.state_size, 3])
        # print('before expand:', states)
        # states = np.expand_dims(states, axis = 0)
        # next_states = np.expand_dims(next_states, axis = 0)
        # print('after expand:', states)

        # Q-Value by state in model
        #target, target_val = [], []
        target = np.zeros((self.batch_size,1,4))
        target_val = np.zeros((self.batch_size,1,4))
        for i in range(len(states)):
            target[i] = self.model.predict(states[i])
            target_val[i] = self.target_model.predict(next_states[i])
            #target.append(temp)
            #target_val.append(temp2)

        # target = self.model.predict(states)
        # target_val = self.target_model.predict(next_states)
        # states shape = {tuple} <class 'tuple'> : (self.batch_size, self.state_size) ; size : self.batch_size * self.state_size
        # states = np.reshape(states, (self.batch_size, -1, 3)) # 64 * 15 * 3
        # next_states = np.reshape(next_states, (self.batch_size, -1, 3))
        # states = np.reshape(states, (1, self.state_size))
        # states = np.expand_dims(states, axis = 0)
        # next_states = np.expand_dims(next_states, axis = 0)
        # target, target_val = [], []
        # for i in range(len(states)):
        #     target.append(self.model.predict(states[i]))

        # Q-Value by next_state in target model
        # for i in range(len(next_states)):
        #     target_val.append(self.model.predict(next_states[i]))

        # Update target by Bellman Optimality Equation
        for i in range(self.batch_size):
            if dones[i]:
                target[i][0][actions[i]] = rewards[i]
            else:
                target[i][0][actions[i]] = rewards[i] + self.discount_factor * np.amax(target_val[i])

        # X = [[0 for x in range(len(states))] for y in range(self.batch_size)]
        # Y = [[0 for x in range(len(target))] for y in range(self.batch_size)]
        #states = np.expand_dims(states, axis=0)
        states = np.squeeze(states)
        target = np.squeeze(target)

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
        # for i in range(self.batch_size):
        #     self.model.fit(states[i], target[i], epochs = 1, verbose = 0)
        # # # self.model.fit(states[0], target[0], batch_size = self.batch_size, epochs = 1, verbose = 1)


# Training
if __name__ == "__main__":
    env = Env()
    state = env.reset()
    # print('state = env.reset()', state)
    state_size = env.state_size

    # Check State
    # print("State_size: ", state_size)

    # Agent
    agent = DQN_Agent(state_size)
    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done, score, step = False, 0, 0

        state = env.reset()
        # print('state = env.reset()', state)
        # state = np.reshape(state, [1, state_size])
        state = np.expand_dims(state, axis = 0)
        # print('before get_action', state)

        total_dead, total_remain_oil = 0, 0

        while not done:
            # if agent.render:
            # env.render()
            global_step += 1
            step += 1

            # Choose Action by state
            action = agent.get_action(state)
            if step > 800:
                action = random.randint(0, 3)

            # Action
            next_state, reward, done, goal, remain_human, remain_oil = env.step(action)

            if step == 1000:
                reward -= 100
                done = True

            if goal == True:
                reward += 100
            # next_state = np.reshape(next_state, [1, state_size])
            next_state = np.expand_dims(next_state, axis = 0)

            # Reward
            # reward = reward if not done or score == 499 else -100

            # <state, action, reward, next_state> in Replay Memory
            agent.append_sample(state, action, reward, next_state, done)

            # if goal == 1:
            # done = True

            # Training by time-step
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            state = next_state
            score += reward

            state = copy.deepcopy(next_state)

            if done:
                agent.update_target_model()

                if goal == 1:
                    result = "Goal!"
                else:
                    result = "No Goal..."

                # Training Result
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'olive')
                pylab.savefig("./save_graph/DQN_Agent.png")
                print("episode:", e, "  score:", score, "step : ", step, "Replay Memory: ", len(agent.memory),
                      "global_step: ", global_step, "  epsilon:", agent.epsilon, "Goal: ", goal, "Total_dead: ", total_dead,
                      "Total_Remain_Oil: ", total_remain_oil)

        # Save model
        if e % 10 == 0:
            agent.model.save_weights("./save_model/DQN_Agent.h5")