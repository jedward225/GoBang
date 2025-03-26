import pygame
import torch
import numpy as np
from 五子棋强化学习AI import GoBangNet, GoBangEnv

class GoBangGame:
    def __init__(self, board_size=15):
        pygame.init()
        self.board_size = board_size
        self.cell_size = 40
        self.margin = 40
        
        # 设置窗口大小
        self.window_size = (board_size * self.cell_size + 2 * self.margin,
                          board_size * self.cell_size + 2 * self.margin)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption("五子棋")
        
        # 加载AI模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ai = GoBangNet(board_size).to(self.device)
        self.ai.load_state_dict(torch.load("gobang_model.pth"))
        self.ai.eval()
        
        # 初始化环境
        self.env = GoBangEnv(board_size)
        
    def draw_board(self):
        """绘制棋盘"""
        self.screen.fill((255, 255, 255))
        
        # 绘制网格线
        for i in range(self.board_size):
            # 横线
            pygame.draw.line(self.screen, (0, 0, 0),
                           (self.margin, self.margin + i * self.cell_size),
                           (self.margin + (self.board_size-1) * self.cell_size,
                            self.margin + i * self.cell_size))
            # 竖线
            pygame.draw.line(self.screen, (0, 0, 0),
                           (self.margin + i * self.cell_size, self.margin),
                           (self.margin + i * self.cell_size,
                            self.margin + (self.board_size-1) * self.cell_size))
        
        # 绘制棋子
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.env.board[i][j] == 1:  # 黑棋
                    pygame.draw.circle(self.screen, (0, 0, 0),
                                    (self.margin + j * self.cell_size,
                                     self.margin + i * self.cell_size), 15)
                elif self.env.board[i][j] == -1:  # 白棋
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                    (self.margin + j * self.cell_size,
                                     self.margin + i * self.cell_size), 15)
                    pygame.draw.circle(self.screen, (0, 0, 0),
                                    (self.margin + j * self.cell_size,
                                     self.margin + i * self.cell_size), 15, 1)
    
    def get_position(self, pos):
        """将鼠标位置转换为棋盘坐标"""
        x = round((pos[1] - self.margin) / self.cell_size)
        y = round((pos[0] - self.margin) / self.cell_size)
        if 0 <= x < self.board_size and 0 <= y < self.board_size:
            return x, y
        return None
    
    def ai_move(self):
        """AI下棋"""
        state = self.env._get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy, _ = self.ai(state_tensor)
            policy = policy.exp().view(self.board_size, self.board_size)
            
            # 将已有棋子的位置概率设为0
            valid_moves = (self.env.board == 0)
            policy = policy * torch.FloatTensor(valid_moves).to(self.device)
            
            # 选择最高概率的动作
            move = policy.argmax().item()
            return (move // self.board_size, move % self.board_size)
    
    def run(self):
        """运行游戏"""
        running = True
        game_over = False
        
        while running:
            self.draw_board()
            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    
                if not game_over and event.type == pygame.MOUSEBUTTONDOWN:
                    pos = self.get_position(event.pos)
                    if pos is not None:
                        # 玩家下棋
                        next_state, reward, done = self.env.step(pos)
                        if done:
                            if reward == 1:
                                print("玩家胜利！")
                            elif reward == -1:
                                print("无效移动！")
                            game_over = True
                        else:
                            # AI下棋
                            ai_pos = self.ai_move()
                            next_state, reward, done = self.env.step(ai_pos)
                            if done:
                                if reward == 1:
                                    print("AI胜利！")
                                elif reward == 0:
                                    print("平局！")
                                game_over = True
            
            if game_over:
                # 等待玩家关闭窗口或重新开始
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        # 按R键重新开始
                        self.env.reset()
                        game_over = False

if __name__ == "__main__":
    game = GoBangGame()
    game.run() 