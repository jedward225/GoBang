import random
import pygame
import string

class InputBox:
    def __init__(self,surf,left,top,width,height,font):
        self.surf = surf
        self.font = font
        self.rect = pygame.Rect(left,top,width,height)
        self.list = []
        self.active = False
        self.cursor = True
        self.count = 0
        self.delete = False

    def draw(self):
        pygame.draw.rect(self.surf,(0,0,0),self.rect,1)
        text_pic =  self.font.render(''.join(self.list),True,(0,0,0))
        self.surf.blit(text_pic,(self.rect.x+5,self.rect.y+10))
        self.count += 1
        if self.count == 20:
            self.count = 0
            self.cursor = not self.cursor

        if self.active and self.cursor:
            text_pic_rect = text_pic.get_rect()
            x = self.rect.x+5+text_pic_rect.width
            pygame.draw.line(self.surf,(0,0,0),(x,self.rect.y+5),(x,self.rect.y+self.rect.height-5),1)

        if self.delete and self.list:
            clock=pygame.time.Clock()
            clock.tick(12)
            self.list.pop()
            
    def get_text(self,event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False

        elif self.active:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:    
                    self.delete = True
                elif event.unicode in string.ascii_letters or event.unicode in '1234567890_':
                    self.list.append(event.unicode)

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_BACKSPACE:
                    self.delete = False

    @property
    def text(self):
        return ''.join(self.list)

def setchess(x,y,board,side):
    if side == 1:
        if board[x][y] == ' ' :
            board[x][y] = 'X'
            return 0
        elif board[x][y] != ' ' :
            return -1
    elif side == 0:
        if board[x][y] == ' ' :
            board[x][y] = 'O'
            return 0
        elif board[x][y] != ' ' :
            return -1

def check(a,board):
    board2 = list(map(list, zip(*board)))
    board3 = [[] for line in range(2*a-1)]
    for x in range(a):
        for y in range(a):
            board3[x-y+a-1].append(board[x][y])
    board4 = [[] for line in range(a*2-1)]
    for x in range(a):
        for y in range(a):
            board4[y+x].append(board[x][y])
               
    Q = [board,board2,board3,board4]  
    for i in Q:
        for line in i:
            if ''.join(line).find('X'*5) != -1:
                return 1
            elif ''.join(line).find('O'*5) != -1:
                return 0 
    return -1

def choose():
    clock = pygame.time.Clock()
    pygame.init()
    screen = pygame.display.set_mode((400,200))
    pygame.display.set_caption("五子棋")
    font0 = pygame.font.SysFont('SimSun', 20)
    len = font0.render('输入棋盘边长:', True, (0,0,0)) 
    box = InputBox(screen,200,75,100,40,font0)
    while True:
        clock.tick(30)
        screen.fill((255,255,255))
        screen.blit(len,(50,85))
        box.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            else:
                box.get_text(event)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return box.text
        pygame.display.update()

def main(): 
    a = int(choose())
    board=[[' ']*a for line in range(a)]
    pygame.init()
    window = pygame.display.set_mode((a*40+40,a*40+40))
    pygame.display.set_caption("五子棋")
    window.fill((255,255,255))
    x, y = 0, 0
    dy, dx = 40, 40
    i = 0
    while i<a: 
        pygame.draw.line(window,(0,0,0),(40,y+dy),(a*40,y+dy))
        pygame.draw.line(window,(0,0,0),(x+dx,40),(x+dx,a*40))
        y += dy
        x += dx
        i += 1
        pygame.display.flip()
    black = pygame.image.load('images/black.png')
    white = pygame.image.load('images/white.png')
    
    side = random.randint(0,1)

    text0 = ['白先','黑先']
    font = pygame.font.SysFont('SimSun', 24)

    if side == 1:
        text = font.render(text0[1],True,(0,0,0)) 
    else:
        text = font.render(text0[0],True,(0,0,0)) 
    
    window.blit(text,(0,0))
    end = True
    color = [white,black]
    count = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN and end:
                p = pygame.mouse.get_pos()
                x, y = round(p[0]/40-1), round(p[1]/40-1)
                checker = 1
                if x<0 or x>=a or y<0 or y>=a:
                    checker = -1
                if checker != -1 :
                    checker = setchess(x,y,board,side)
                if checker != -1 :    
                    window.blit(color[side],(x*40+20,y*40+20))
                    setchess(x,y,board,side)
                    side = 1 - side
                    count += 1
            checkwin=check(a,board)
            if checkwin == 1:
                text1 = font.render('黑棋胜！',True,(255,0,0)) 
            elif checkwin == 0:
                text1 = font.render('白棋胜！',True,(255,0,0))
            elif count==a**2:
                text1 = font.render('平局',True,(255,0,0))
                checkwin = 3
            if checkwin != -1:
                window.blit(text1,(100,0))
                end = False
               
        pygame.display.update()
   
if __name__ == '__main__':
    main()
