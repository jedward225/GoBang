import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import pygame
import copy

class GoBangEnv:
    """五子棋环境"""
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.reset()
        
    def reset(self):
        """重置棋盘"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1为黑棋，-1为白棋
        self.done = False
        return self._get_state()
    
    def _get_state(self):
        """获取当前状态，转换为神经网络输入格式"""
        # 3个通道：己方棋子、对方棋子、当前手
        state = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        state[0] = (self.board == self.current_player)
        state[1] = (self.board == -self.current_player)
        state[2] = np.ones((self.board_size, self.board_size)) * (self.current_player == 1)
        return state
    
    def step(self, action):
        """执行一步动作
        action: (x, y) 坐标元组
        returns: (next_state, reward, done)
        """
        x, y = action
        
        # 检查动作是否合法
        if self.board[x][y] != 0:
            return self._get_state(), -1, True
        
        # 落子
        self.board[x][y] = self.current_player
        
        # 检查是否获胜
        if self._check_win(x, y):
            return self._get_state(), 1, True
            
        # 检查是否平局
        if np.count_nonzero(self.board) == self.board_size * self.board_size:
            return self._get_state(), 0, True
            
        # 切换玩家
        self.current_player *= -1
        return self._get_state(), 0, False
        
    def _check_win(self, x, y):
        """检查是否获胜"""
        player = self.board[x][y]
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        
        for dx, dy in directions:
            count = 1
            # 正向检查
            for i in range(1, 5):
                new_x, new_y = x + dx*i, y + dy*i
                if not (0 <= new_x < self.board_size and 0 <= new_y < self.board_size):
                    break
                if self.board[new_x][new_y] != player:
                    break
                count += 1
            # 反向检查
            for i in range(1, 5):
                new_x, new_y = x - dx*i, y - dy*i
                if not (0 <= new_x < self.board_size and 0 <= new_y < self.board_size):
                    break
                if self.board[new_x][new_y] != player:
                    break
                count += 1
            if count >= 5:
                return True
        return False

class GoBangNet(nn.Module):
    """五子棋策略价值网络"""
    def __init__(self, board_size=15):
        super(GoBangNet, self).__init__()
        # 共享特征提取层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 策略头
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)
        
        # 价值头
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def forward(self, x):
        # 特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # 策略头
        policy = F.relu(self.policy_conv(x))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.log_softmax(policy, dim=1)
        
        # 价值头
        value = F.relu(self.value_conv(x))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class GoBangAI:
    """五子棋AI代理"""
    def __init__(self, board_size=15):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.board_size = board_size
        self.policy_value_net = GoBangNet(board_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer(10000)
        
    def select_action(self, state, epsilon=0.1):
        """选择动作，使用epsilon-贪婪策略"""
        if random.random() < epsilon:
            # 随机探索
            valid_moves = np.where(state[0] == 0)
            move_idx = random.randint(0, len(valid_moves[0])-1)
            return (valid_moves[0][move_idx], valid_moves[1][move_idx])
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            policy, value = self.policy_value_net(state_tensor)
            policy = policy.exp().view(self.board_size, self.board_size)
            
            # 将已经有棋子的位置的概率设为0
            valid_moves = (state[0] == 0)
            policy = policy * torch.FloatTensor(valid_moves).to(self.device)
            
            # 选择最高概率的动作
            move = policy.argmax().item()
            return (move // self.board_size, move % self.board_size)
    
    def _action_to_index(self, action):
        """将动作坐标转换为一维索引"""
        x, y = action
        return x * self.board_size + y
    
    def train(self, batch_size=32, gamma=0.99):
        """训练网络"""
        if len(self.replay_buffer) < batch_size:
            return
            
        # 采样mini-batch
        transitions = self.replay_buffer.sample(batch_size)
        batch = list(zip(*transitions))
        
        # 准备数据
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        # 将动作转换为一维索引
        action_indices = [self._action_to_index(action) for action in batch[1]]
        action_batch = torch.LongTensor(action_indices).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # 计算当前Q值
        policy, value = self.policy_value_net(state_batch)
        
        # 计算目标Q值
        next_policy, next_value = self.policy_value_net(next_state_batch)
        target_value = reward_batch + gamma * next_value.squeeze() * (1 - done_batch)
        
        # 计算损失
        value_loss = F.mse_loss(value.squeeze(), target_value)
        policy_loss = F.nll_loss(policy, action_batch)
        total_loss = value_loss + policy_loss
        
        # 优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()

def evaluate_ai(ai, num_games=10):
    """评估AI的性能"""
    env = GoBangEnv()
    wins = 0
    draws = 0
    
    for game in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            action = ai.select_action(state, epsilon=0.0)  # 评估时不使用探索
            state, reward, done = env.step(action)
            
            if done:
                if reward == 1:
                    wins += 1
                elif reward == 0:
                    draws += 1
                break
    
    return wins, draws, num_games - wins - draws

def self_play_training(ai, num_games=100):
    """自我对弈训练"""
    env = GoBangEnv()
    
    for game in range(num_games):
        state = env.reset()
        game_memory = []
        done = False
        
        while not done:
            # 选择动作
            action = ai.select_action(state, epsilon=0.1)
            next_state, reward, done = env.step(action)
            
            # 存储经验
            game_memory.append((state, action, reward))
            
            if done:
                # 根据游戏结果调整奖励
                final_reward = reward
                # 反向传播奖励
                for prev_state, prev_action, _ in reversed(game_memory):
                    ai.replay_buffer.push(prev_state, prev_action, final_reward, next_state, done)
                    final_reward *= 0.95  # 衰减奖励
            
            state = next_state
            
            # 训练
            if len(ai.replay_buffer) >= 32:
                loss = ai.train()
                
        if game % 10 == 0:
            wins, draws, losses = evaluate_ai(ai, num_games=5)
            print(f"自我对弈游戏 {game}, 评估结果 - 胜: {wins}, 平: {draws}, 负: {losses}")

def train_ai():
    """训练AI"""
    try:
        print("初始化环境和AI...")
        ai = GoBangAI()
        
        print("开始自我对弈训练...")
        for iteration in range(10):  # 10轮训练
            print(f"训练轮次 {iteration + 1}/10")
            self_play_training(ai, num_games=100)  # 每轮100局自我对弈
            
            # 保存模型检查点
            print(f"保存模型检查点 {iteration}...")
            torch.save(ai.policy_value_net.state_dict(), f"gobang_model_iter{iteration}.pth")
            
            # 评估
            wins, draws, losses = evaluate_ai(ai, num_games=20)
            print(f"轮次 {iteration + 1} 评估结果 - 胜: {wins}, 平: {draws}, 负: {losses}")
        
        return ai
        
    except Exception as e:
        print(f"训练过程出错: {e}")
        raise

if __name__ == "__main__":
    try:
        print("开始五子棋AI训练程序...")
        ai = train_ai()
        print("训练完成，保存最终模型...")
        torch.save(ai.policy_value_net.state_dict(), "gobang_model_final.pth")
        
        # 最终评估
        wins, draws, losses = evaluate_ai(ai, num_games=50)
        print(f"最终评估结果 - 胜: {wins}, 平: {draws}, 负: {losses}")
        print("程序结束")
    except Exception as e:
        print(f"程序执行出错: {e}")
        raise