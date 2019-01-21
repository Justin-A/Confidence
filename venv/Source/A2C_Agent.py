import matplotlib
matplotlib.use('TkAgg')

import copy
import pylab
import numpy as np
from environment import Env
from keras.layers import Dense, Conv1D
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K
import random

EPISODES = 1000

class A2CAgent:
    def __init__(self, state_size):
        self.load_model = False
        # 에이전트가 가능한 모든 행동 정의
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.discount_factor = 0.99
        self.action_space = ['u', 'd', 'l', 'r']
        self.action_size = len(self.action_space)
        self.value_size = 1

        # 액터-크리틱 하이퍼파라미터
        self.discount_factor = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.0005

        # 정책신경망과 가치신경망 생성
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_updater = self.actor_optimizer()
        self.critic_updater = self.critic_optimizer()

        if self.load_model:
            self.actor.load_weights('./save_model/grid_actor.h5')
            self.critic.load_weights('./save_model/grid_critic.h5')

    # actor: 상태를 받아 각 행동의 확률을 계산
    def build_actor(self):
        actor = Sequential()
        actor.add(Conv1D(filters=10, kernel_size=1,
                         input_shape=(self.state_size, 3), kernel_initializer='he_uniform'))
        actor.add(Dense(30, activation='tanh', kernel_initializer='he_uniform'))
        actor.add(Dense(30, activation='tanh'))
        actor.add(Dense(self.action_size, activation='softmax'))
        actor.summary()
        return actor

    # critic: 상태를 받아서 상태의 가치를 계산
    def build_critic(self):
        critic = Sequential()
        critic.add(Conv1D(filters=10, kernel_size=self.state_size,
                          input_shape=(self.state_size, 3), kernel_initializer='he_uniform'))
        critic.add(Dense(24, input_dim=self.state_size, activation='relu'))
        critic.add(Dense(24, activation='relu'))
        critic.add(Dense(self.value_size, activation='liㄴnear'))
        critic.summary()
        return critic

    # 정책신경망의 출력을 받아 확률적으로 행동을 선택
    def get_action(self, state):
        policy = self.actor.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0], policy

    # 정책신경망을 업데이트하는 함수
    def actor_optimizer(self):
        action = K.placeholder(shape=[None, self.action_size])
        advantage = K.placeholder(shape=[None, ])

        action_prob = K.sum(action * self.actor.output, axis=1)
        cross_entropy = K.log(action_prob + 1e-10) * advantage
        cross_entropy = -K.sum(cross_entropy)

        entropy = K.sum(self.actor.output * K.log(self.actor.output + 1e-10), axis=1)
        loss = cross_entropy + 0.01 * entropy

        optimizer = Adam(lr=self.actor_lr)
        updates = optimizer.get_updates(self.actor.trainable_weights, [], loss)
        train = K.function([self.actor.input, action, advantage], [],
                           updates=updates)
        return train

    # 가치신경망을 업데이트하는 함수
    def critic_optimizer(self):
        target = K.placeholder(shape=[None, ])

        loss = K.mean(K.square(target - self.critic.output))

        optimizer = Adam(lr=self.critic_lr)
        updates = optimizer.get_updates(self.critic.trainable_weights, [], loss)
        train = K.function([self.critic.input, target], [], updates=updates)

        return train

    # 각 타임스텝마다 정책신경망과 가치신경망을 업데이트
    def train_model(self, state, action, reward, next_state, done):
        value = self.critic.predict(state)[0]
        next_value = self.critic.predict(next_state)[0]

        act = np.zeros([1, self.action_size])
        act[0][action] = 1

        # 벨만 기대 방정식를 이용한 어드벤티지와 업데이트 타깃
        if done:
            advantage = reward - value
            target = [reward]
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        self.actor_updater([state, act, advantage])
        self.critic_updater([state, target])

if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env()
    state = env.reset()
    state_size = env.state_size
    print('state size : ', state_size)
    agent = A2CAgent(state_size)

    global_step = 0
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        step = 0
        state = env.reset()

        #state = np.reshape(state, [1, agent.state_size])
        #print(state)
        state = np.expand_dims(state, axis = 0)
        #print(state)
        goal = 0
        total_dead, total_remain_oil = 0, 0

        while not done:
            # env 초기화
            global_step += 1
            step += 1

            #rint(state)
            # 현재 상태에 대한 행동 선택
            action, policy = agent.get_action(state)
            if global_step == 1:
                policy = [0.25, 0.25, 0.25, 0.25]

            # if e < 20:
            #     action = random.randint(0, 3)

            if step > 500:
                action = random.randint(0, 3)

            next_state, reward, done, goal, remain_human, remain_oil = env.step(action)
            #next_state = np.reshape(next_state, [1, agent.state_size])

            if step % 100 == 0:
                value = agent.critic.predict(state)[0]
                next_state_tmp = np.expand_dims(next_state, axis=0)
                next_value = agent.critic.predict(next_state_tmp)[0]
                advantage = (reward + agent.discount_factor * next_value) - value
                print('policy : ', policy, 'action : ', action, 'advantage : ', advantage)

            next_state = np.expand_dims(next_state, axis=0)
            agent.train_model(state, action, reward, next_state, done)

            state = next_state
            score += reward

            state = copy.deepcopy(next_state)

            if done:
                # 에피소드마다 학습 결과 출력
                if goal == 1:
                    goal = "Goal!"
                else:
                    goal = "No Goal…"
                    total_dead += remain_human
                    total_remain_oil += remain_oil

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("./save_graph/a2c.png")
                print("episode:", e, "  score:", score, "step : ", step, "global_step: ", global_step, "Goal : ", goal, "Total_dead: ", total_dead,
                      "Total_Remain_Oil: ", total_remain_oil)

        # 100 에피소드마다 모델 저장
        if e % 100 == 0:
            agent.actor.save_weights("./save_model/grid_actor.h5")
            agent.critic.save_weights("./save_model/grid_critic.h5")