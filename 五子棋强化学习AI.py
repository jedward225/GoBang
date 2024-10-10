import numpy as np
from collections import deque

class GomokuEnv:
    def __init__(self, board_size):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1: 黑棋，-1: 白棋
        return self.board, self.current_player

    def step(self, action):
        if not self.is_valid_action(action):
            print(f"Invalid action {action} received. Skipping this step.")
            return self.board, 0, False, {}  # 返回当前棋盘状态、奖励为0、游戏未结束和空字典

        assert self.is_valid_action(action)
        x, y = action
        self.board[x, y] = self.current_player

        # 检查棋盘是否已满，如果是，则宣布平局并返回
        if np.all(self.board != 0):
            print("Board is full. Declaring a draw.")
            reward = 0  # 平局奖励为0
            done = True
        else:
            winner = self.check_win()  # 检查是否有玩家获胜或平局
            reward = self.get_reward(winner)
            done = winner is not None  # 如果游戏结束，则设置done为True

        self.current_player *= -1  # 切换当前玩家
        return self.board, reward, done, {}

    def is_valid_action(self, action):
        x, y = action
        return self.board[x, y] == 0  # 位置为空

    def check_win(self):
        # 实现棋盘状态检查，判断是否有玩家获胜或平局
        pass

    def get_reward(self, winner):
        if winner == 1:
            return 1  # 黑棋获胜
        elif winner == -1:
            return -1  # 白棋获胜
        else:
            return 0  # 平局或游戏未结束

# 定义Q-learning训练器类
class QLearningTrainer:
    def __init__(self, env, learning_rate, discount_factor, epsilon, exploration_decay, replay_memory_size):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.exploration_decay = exploration_decay
        self.replay_memory = deque(maxlen=replay_memory_size)

    def train(self, num_episodes, max_steps_per_episode):
        q_table = np.zeros((self.env.board_size, self.env.board_size, self.env.board_size, self.env.board_size))

        for episode in range(num_episodes):
            state, player = self.env.reset()
            for _ in range(max_steps_per_episode):
                action = self.select_action(state, player, q_table)
                if not self.env.is_valid_action(action):  # 新增：检查选择的action是否有效
                    print(f"Invalid action {action} returned by select_action. Skipping this step.")
                    continue  # 跳过本次循环，避免将无效经验添加到重放缓冲区

                next_state, reward, done, _ = self.env.step(action)

                # 将经验存储到重放缓冲区
                self.replay_memory.append((state, player, action, reward, next_state, done))
                if len(self.replay_memory) > self.replay_memory.maxlen // 2:
                    # 从重放缓冲区中采样经验进行学习
                    batch = np.random.choice(self.replay_memory, size=32, replace=False)
                    states, players, actions, rewards, next_states, dones = zip(*batch)

                    # 更新Q表
                    for s, p, a, r, ns, d in zip(states, players, actions, rewards, next_states, dones):
                        q_target = r
                        if not d:
                            best_next_action = self.get_best_action(ns, p, q_table)
                            q_target = r + self.discount_factor * q_table[tuple(best_next_action)]

                        q_table[s[0], s[1], s[2], s[3]][a] += self.learning_rate * (q_target - q_table[s[0], s[1], s[2], s[3]][a])

                if done:
                    break

                state = next_state
                player = -player  # 切换玩家

            # 衰减探索率
            self.epsilon *= self.exploration_decay

        return q_table

    def select_action(self, state, player, q_table):
        valid_actions = []
        for i in range(self.env.board_size):
            for j in range(self.env.board_size):
                if self.env.is_valid_action((i, j)):  # 检查该位置是否为空
                    valid_actions.append((i, j))

        if not valid_actions:
            print("Warning: No valid actions found! Returning a default action.")
            return (0, 0)  # 返回一个默认动作，这里以(0, 0)为例

        if np.random.rand() < self.epsilon:
            # 随机选择一个有效的动作
            action = valid_actions[np.random.randint(len(valid_actions))]
        else:
            # 基于Q表选择最佳动作，仅在有效动作中选择
            max_q_values = [q_table[state[0], state[1], i, j] for i, j in valid_actions]
            best_action_idx = np.argmax(max_q_values)
            action = valid_actions[best_action_idx]

        return action
    def get_best_action(self, state, player, q_table):
        return np.unravel_index(np.argmax(q_table[state[0], state[1], :, :]), (self.env.board_size, self.env.board_size))

# 训练五子棋AI
board_size = 15
num_episodes = 10000
max_steps_per_episode = board_size ** 2 + 1  # 适当设置最大步数

env = GomokuEnv(board_size)
trainer = QLearningTrainer(env, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, exploration_decay=0.999,
                           replay_memory_size=10000)
q_table = trainer.train(num_episodes, max_steps_per_episode)

# 保存训练好的Q表以供后续使用
np.save("gomoku_q_table.npy", q_table)