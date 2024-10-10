import random
import pygame
import string
import copy
import time
value_black1 = {
    1_1 : ('    XO',3),
    1_2 : ('OX    ',3),
    1_3 : ('    X    ',1),
    2_1 : ('   XX   ',8),
    2_2 : ('OXX   ',6),
    2_3 : ('   XXO',6),
    2_4 : (' XX   ',7),
    2_5 : (' X X ',6),
    2_6 : ('XX   ',6),
    2_7 : ('   XX',6)
}
value_black2 = {
    3_1 : ('  XXX  ',9),
    3_2 : (' XX X ',30),
    3_3 : (' X XX  ',2),
    3_4 : (' X XX ',28),
    3_5 : ('XXX  ',26),
    3_6 : ('  XXX' ,26),
    3_7 : ('OXX X ',21),
    3_8: (' X XXO',21),
} 
value_black3 = {
    4_1 : (' XXXX ',800),
    4_2 : (' XXXXO',240),
    4_3 : ('OXXXX ',200),
    4_4 : ('X XXX  ',188),
    4_5 : ('   XX XX   ',194),
    4_6 : ('  XXX X',240),
    4_7 : ('X XXXO',164),
    4_8 : ('XX XXO',164),
    4_9 : ('OXX XX',170),
    4_10: ('OXXX X',170),
    4_11: ('XXXX ',200),
    4_12: ('XX XX',170),
    4_13: ('XXX X',170),
    4_14: ('XXX XO',164),
    4_15: ('X XXX',170),
    4_16: (' XXXXO',164),

    5_1 : ('XXXXX',10000)
}
value_white1 = {
    1_1 : ('XO    ',3),
    1_2 : ('    OX',3),
    1_3 : ('    O    ',1),
    2_1 : ('   OO   ',8),
    2_2 : ('XOO   ',6),
    2_3 : ('   OOX',6),
    2_4 : (' OO   ',7),
    2_5 : (' O O ',6),
    2_6 : ('OO   ',6),
    2_7 : ('   OO',6)
}
value_white2 = {
    3_1 : ('  OOO  ',9),
    3_2 : (' OO O ',30),
    3_3 : (' O OO  ',2),
    3_4 : (' O OO ',28),
    3_5 : ('OOO  ',26),
    3_6 : ('  OOO' ,26),
    3_7 : ('XOO O ',21),
    3_8: (' O OOX',21),
}
value_white3 = {
    4_1 : (' OOOO ',800),
    4_2 : (' OOOOX',240),
    4_3 : ('XOOOO ',200),
    4_4 : ('O OOO  ',188),
    4_5 : ('   OO OO   ',194),
    4_6 : ('  OOO O',240),
    4_7 : ('O OOOX',164),
    4_8 : ('OO OOO',164),
    4_9 : ('XOO OO',170),
    4_10: ('XOOO O',170),
    4_11: ('OOOO ',200),
    4_12: ('OO OO',164),
    4_13: ('OOO O',164),
    4_14: ('OOO OO',170),
    4_15: ('O OOO',170),
    4_16: (' OOOO',164),

    5_1 : ('OOOOO',10000)
}
huosan_B = {
    1_1:' XXX ',
    1_2:' X XX ',
    1_3:' XX X ',
}
huosan_W = {
    1_1:' OOO ',
    1_2:' O OO ',
    1_3:' OO O ',
}
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
                elif event.unicode in string.ascii_letters or event.unicode in '1234567890':
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
        elif board[x][y] != ' ':
            return -1

def check(a,board):
    for line in board:
        if ''.join(line).find('X'*5) != -1:
            return 1
        elif ''.join(line).find('O'*5) != -1:
            return 0
    board2 = list(map(list, zip(*board)))
    for line in board2:
        if ''.join(line).find('X'*5) != -1:
            return 1
        elif ''.join(line).find('O'*5) != -1:
            return 0
    board3 = [[] for line in range(2*a-1)]
    for x in range(a):
        for y in range(a):
            board3[x-y+a-1].append(board[x][y])
    for line in board3:
        if ''.join(line).find('X'*5) != -1:
            return 1
        elif ''.join(line).find('O'*5) != -1:
            return 0
    board4 = [[] for line in range(a*2-1)]
    for x in range(a):
        for y in range(a):
            board4[y+x].append(board[x][y])
        for line in board4:
            if ''.join(line).find('X'*5) != -1:
                return 1
            elif ''.join(line).find('O'*5) != -1:
                return 0          
    return -1

def ban_of_hand(x,y,a,board,side):
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
    b, w = 0, 0
    for i in Q:
        for line in i:
            if side == 1:
                for x in huosan_B.items():
                    if ''.join(line).find(str(x[1]))!=-1:
                        b += 1
            else:       
                for x in huosan_W.items():
                    if ''.join(line).find(str(x[1]))!=-1:
                        w += 1
    if side == 1 and b > 1:
        return False
    if side == 0 and w > 1:    
        return False          
    return True

def check_value(a,board):

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
    B1,B2,B3 = 0,0,0
    W1,W2,W3 = 0,0,0
    cb, cw = 0, 0
    for i in Q:
        for line in i:
            for x in value_black1.items():
                if ''.join(line).find(str(x[1][0]))!=-1:
                    B1 += int(x[1][1])
            for x in value_white1.items():
                if ''.join(line).find(str(x[1][0]))!=-1:
                    W1 += int(x[1][1])
            for x in value_black2.items():
                if ''.join(line).find(str(x[1][0]))!=-1:
                    B2 += int(x[1][1])
                    cb += 1
            for x in value_white2.items():
                if ''.join(line).find(str(x[1][0]))!=-1:
                    W2 += int(x[1][1])
                    cw += 1
            for x in value_black3.items():
                if ''.join(line).find(str(x[1][0]))!=-1:
                    B3 += int(x[1][1])
            for x in value_white3.items():
                if ''.join(line).find(str(x[1][0]))!=-1:
                    W3 += int(x[1][1])
    if cw > 1:
        W2 *= 4
    if cb > 1:
        B2 *= 4
    B = B1 + B2 + B3
    W = W1 + W2 + W3
    return B,W

def choose():
    clock=pygame.time.Clock()
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

def AIsetchess(a,AI,board,side):
    valueW, valueB = [], []
    for x in range(a):
        for y in range(a):
            if side == AI:
                board_copy = copy.deepcopy(board)
            BOARD = copy.deepcopy(board)
            BOARD2 = copy.deepcopy(board)
            if BOARD[x][y] == ' ':
                setchess(x,y,BOARD,AI)
                b1,w1 = check_value(a,BOARD)
                setchess(x,y,BOARD2,1-AI)
                b2,w2 = check_value(a,BOARD2)
                if side == AI:
                    if ban_of_hand(x,y,a,BOARD,AI) == False:
                        b1,b2,w1,w2 = 0,0,0,0
                if AI == 1:
                    valueB.append(b1)
                    valueW.append(w2)
                elif AI == 0:
                    valueB.append(b2)
                    valueW.append(w1)
            else : 
                valueB.append(-1)
                valueW.append(-1)  
    maxw1, maxb1 = max(valueW), max(valueB)
    maxw2, maxb2 = [], [] 
    tag = 0
    for i in valueW:
        if i == maxw1:
            maxw2.append(tag)
        tag += 1
    tag = 0
    for i in valueB:
        if i == maxb1:
            maxb2.append(tag)
        tag += 1
    
    maxw3, maxb3 = 0, 0
    maxb4, maxw4 = [], []
    if maxb4 != [] and maxw4 != []:
        for i in maxb2:
            if maxw3 < valueW[i]:
                maxw3 = valueW[i]
                maxb4.append(i)
        for i in maxw2:
            if maxb3 < valueB[i]:
                maxb3 = valueB[i]
                maxw4.append(i)
        b0 = random.randint(0,len(maxb4)-1)
        w0 = random.randint(0,len(maxw4)-1) 
    else:  
        b0 = random.randint(0,len(maxb2)-1)
        w0 = random.randint(0,len(maxw2)-1)      
    maxb = maxb2[b0]
    maxw = maxw2[w0]
    
    if AI == 1:
        if maxb1 >= 10000:
            loc = maxb
            setchess(loc//a,loc%a,board,AI)
            return loc//a,loc%a
        if maxw1 >= 100 :
            if maxw1 >= 10000 :
                loc = maxw
                setchess(loc//a,loc%a,board,AI)
                return loc//a,loc%a
            if maxb1 <= 160:
                loc = maxw
                setchess(loc//a,loc%a,board,AI)
                return loc//a,loc%a
            if maxb1 > 160 and maxw1 <= 200 :
                loc = maxb
                setchess(loc//a,loc%a,board,AI)
                return loc//a,loc%a 
            if maxb1 > 800: 
                loc = maxb
                setchess(loc//a,loc%a,board,AI)
                return loc//a,loc%a  
            if maxw1 > 200:
                loc = maxw
                setchess(loc//a,loc%a,board,AI)
                return loc//a,loc%a
    else:
        if maxw1 >= 10000:
            loc = maxw
            setchess(loc//a,loc%a,board,AI)
            return loc//a,loc%a
        if maxb1 >= 200:
            if maxb1 >= 10000 :
                loc = maxb
                setchess(loc//a,loc%a,board,AI)
                return loc//a,loc%a
            if maxw1 <= 160:
                loc = maxb
                setchess(loc//a,loc%a,board,AI)
                return loc//a,loc%a
            if maxw1 > 160 and maxb1 <= 200 :
                loc = maxw
                setchess(loc//a,loc%a,board,AI)
                return loc//a,loc%a   
            if maxw1 > 800: 
                loc = maxw
                setchess(loc//a,loc%a,board,AI)
                return loc//a,loc%a 
            if maxb1 > 200:
                loc = maxb
                setchess(loc//a,loc%a,board,AI)
                return loc//a,loc%a
    if AI == 1:
        loc = maxb
        x,y=loc//a,loc%a
    else :
        loc = maxw
        x, y = loc//a,loc%a
    setchess(x,y,board,AI)
    return x,y

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
    first = random.randint(0,1)
    
    text0=['人执白先','人执黑先','AI执白先','AI执黑先']
    font = pygame.font.SysFont('SimSun', 24)

    if side == 1 and first == 1:
        text = font.render(text0[1],True,(0,0,0)) 
    elif side == 0 and first == 1:
        text = font.render(text0[0],True,(0,0,0)) 
    elif side == 1 and first == 0:
        text = font.render(text0[3],True,(0,0,0)) 
    elif side == 0 and first == 0:
        text = font.render(text0[2],True,(0,0,0))
    
    side_flag = side
    window.blit(text,(0,0))
    end = True
    color=[white,black]
    count=0
    pygame.display.update()
    if first == 0:
        time.sleep(1)
        x , y = a//2, a//2
        setchess(x,y,board,side)
        window.blit(color[side],(x*40+20,y*40+20))
        AI = side
        side = 1 - AI
    else :
        AI = 1- side
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            
            elif event.type == pygame.MOUSEBUTTONDOWN and end and side == 1 - AI:
                p = pygame.mouse.get_pos()
                x,y = round(p[0]/40-1), round(p[1]/40-1)
                checker = 1
                if x<0 or x>=a or y<0 or y>=a:
                    checker = -1
                if checker != -1 :
                    board_copy = copy.deepcopy(board)
                    checker = setchess(x,y,board_copy,side)
                if checker != -1 : 
                    if side == side_flag:
                        if ban_of_hand(x,y,a,board_copy,side_flag) == True: 
                            pygame.draw.rect(window,[255,255,255],[100,0,500,23])
                            window.blit(color[side],(x*40+20,y*40+20))
                            setchess(x,y,board,side)
                            side = 1 - side
                            count += 1
                        else:
                            t0 = '下出了禁手，请您重下'
                            t = font.render(t0,True,(0,0,0))
                            window.blit(t,(200,0))
                    else:
                        window.blit(color[side],(x*40+20,y*40+20))
                        setchess(x,y,board,side)
                        side = 1 - side
                        count += 1
            elif side == AI:
                if checkwin == -1:
                    x,y = AIsetchess(a,AI,board,side_flag)
                    window.blit(color[side],(x*40+20,y*40+20))
                side = 1 - side
                count += 1
                
            checkwin = check(a,board)
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
