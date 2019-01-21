# PG
import matplotlib

matplotlib.use('TkAgg')

import copy
import pylab
import random
import numpy as np
from environment import Env
from keras.layers import Dense, Conv1D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K

EPISODES = 1000


class PolicyGradient_Agent:
    def __init__(self, state_size):
        self.load_model = False
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.state_size = state_size
        self.discount_factor = 0.99
        self.learning_rate = 0.001

        self.model = self.build_model()
        self.optimizer = self.optimizer()
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights('./save_model/PolicyGradient_Agent.h5')

    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters = 10, kernel_size = self.state_size, input_shape = (self.state_size, 3), kernel_initializer = 'he_uniform'))
        model.add(Dense(30, activation = 'tanh', kernel_initializer = 'he_uniform'))
        model.add(Dense(30, activation = 'tanh'))
        model.add(Flatten())
        model.add(Dense(self.action_size, activation = 'softmax'))
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

    # Action by Policy
    def get_action(self, state):
        policy = self.model.predict(state)[0]
        return np.random.choice(self.action_size, 1, p = policy)[0]

    # 반환값 계산
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # <state, action, reward> Sample
    def append_sample(self, state, action, reward):
        self.states.append(state[0])
        self.rewards.append(reward)
        act = np.zeros(self.action_size)
        act[action] = 1
        self.actions.append(act)

    # Update Network
    def train_model(self):
        discounted_rewards = np.float32(self.discount_rewards(self.rewards))
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        self.optimizer([self.states, self.actions, discounted_rewards])
        self.states, self.actions, self.rewards = [], [], []


if __name__ == "__main__":
    # 환경과 에이전트의 생성
    env = Env()
    state = env.reset()
    # print('state = env.reset()', state)
    state_size = env.state_size
    print("State_size: ", state_size)
    agent = PolicyGradient_Agent(state_size)

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        step = 0
        state = env.reset()
        # print('state = env.reset()', state)
        state = np.expand_dims(state, axis = 0)
        # print('np.expand_dims(state, axis = 0', state)
        # state = np.reshape(state, [1, state_size])

        while not done:
            global_step += 1
            step += 1
            # print('before get_action', state)
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

            # next_state = np.reshape(next_state, [1, state_size])
            next_state = np.expand_dims(next_state, axis = 0)

            agent.append_sample(state, action, reward)
            score += reward
            state = copy.deepcopy(next_state)

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
                print("episode: ", e, " score: ", score, "global_step: ", global_step, "goal: ", goal)

        # Save Weight
        if e % 100 == 0:
            agent.model.save_weights("./save_model/PolicyGradient_Agent.h5")
