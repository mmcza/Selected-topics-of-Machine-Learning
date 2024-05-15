import gym
import time
import numpy as np
def main():
    env = gym.make('CartPole-v1', render_mode='human')  # utworzenie środowiska
    env.reset()  # reset środowiska do stanu początkowego
    for _ in range(1000):  # kolejne kroki symulacji
        env.render()  # renderowanie obrazu
        action = env.action_space.sample()  # wybór akcji (tutaj: losowa akcja)
        env.step(action)  # wykonanie akcji
        time.sleep(0.5)
        print(env.step(action)[0][2])
        print(env.step(action)[1])
        # if env.step(action)[2]:
        #     env.reset()
    env.close()  # zamknięcie środowiska

def frozen_lake():
    env = gym.make('FrozenLake-v1', render_mode=None, desc=None, map_name="4x4", is_slippery = False)  # utworzenie środowiska
    env.reset()  # reset środowiska do stanu początkowego

    # implement the Q-learning algorithm
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    alpha = 0.5
    gamma = 0.9
    exploration_rate = 0.5


    n = 100000
    for i in range(n):

        state = env.reset()
        state = state[0]
        done = False
        if i % 1000 == 0:
            print(i)
        while not done:
            exploration_or_exploitation = np.random.rand()
            if exploration_or_exploitation < (exploration_rate-((exploration_rate/n)*i)):
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            next_state, reward, done, _, _ = env.step(action)
            Q[state, action] = (1-alpha)*Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]))
            state = next_state

    env = gym.make('FrozenLake-v1', render_mode='human', desc=None, map_name="4x4",
                   is_slippery=False)  # utworzenie środowiska
    print(Q)

    for i in range(10):
        state = env.reset()
        env.render()
        state = state[0]
        done = False
        while not done:
            action = np.argmax(Q[state, :])
            next_state, reward, done, _, _ = env.step(action)
            state = next_state
            if done:
                print(reward)
                break


if __name__ == '__main__':
    #main()
    frozen_lake()