import numpy as np
import random
import pygame
import sys
import math

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255,255,255)
DARK_BLUE = (44, 62, 80)
NEON_GREEN = (57, 255, 20)
SOFT_ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)

ROW_COUNT = 6
COLUMN_COUNT = 7

PLAYER = 0
AI = 1

EMPTY = 0
PLAYER_PIECE = 1
AI_PIECE = 2

WINDOW_LENGTH = 4

# create 2d array for board
def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board

# drop the coin in particular column and row
def drop_piece(board, row, col, piece):
    board[row][col] = piece

# check if the top of board(i.e last row) is empty so that coin can be dropped
def is_valid_location(board, col):
    return board[ROW_COUNT - 1][col] == 0

# get row where coin would fall
def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r

# print the inverted 2d array
def print_board(board):
    print(np.flip(board, 0))

# check if player wins i.e four coins in a line
def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True, [(r, c), (r, c + 1), (r, c + 2), (r, c + 3)]

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True, [(r, c), (r + 1, c), (r + 2, c), (r + 3, c)]

    # Check positively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True, [(r, c), (r + 1, c + 1), (r + 2, c + 2), (r + 3, c + 3)]

    # Check negatively sloped diagonals
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True, [(r, c), (r - 1, c + 1), (r - 2, c + 2), (r - 3, c + 3)]

    return False, []


# get score of an array of length 4
def evaluate_window(window, piece):
    score = 0
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = AI_PIECE

    if window.count(piece) == 4:
        score += 100
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

# get score
def score_position(board, piece):
    score = 0

    ## Score center column
    center_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    center_count = center_array.count(piece)
    score += center_count * 3

    ## Score Horizontal
    for r in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[r, :])]
        for c in range(COLUMN_COUNT - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score Vertical
    for c in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, c])]
        for r in range(ROW_COUNT - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    ## Score posiive sloped diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + 3 - i][c + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score

# check if player or ai wins or board is full
def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE)[0] or winning_move(board, AI_PIECE)[0] or len(get_valid_locations(board)) == 0


def minimax(board, depth, alpha, beta, maximizingPlayer):
    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, AI_PIECE)[0]:
                return (None, 100000000000000)
            elif winning_move(board, PLAYER_PIECE)[0]:
                return (None, -10000000000000)
            else:  # Game is over, no more valid moves
                return (None, 0)
        else:  # Depth is zero
            return (None, score_position(board, AI_PIECE))
    if maximizingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, AI_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else:  # Minimizing player
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

# get columns where coins can be dropped (i.e columns which are not completely filled)
def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def pick_best_move(board, piece):
    valid_locations = get_valid_locations(board)
    best_score = -10000
    best_col = random.choice(valid_locations)
    for col in valid_locations:
        row = get_next_open_row(board, col)
        temp_board = board.copy()
        drop_piece(temp_board, row, col, piece)
        score = score_position(temp_board, piece)
        if score > best_score:
            best_score = score
            best_col = col

    return best_col


def draw_board(board):
    # draw the board with holes
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
            int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    if(two_player):
        color1=P2_color
        color2=P1_color
    else:
        color1=player_color
        color2=ai_color

    # draw the coins in board
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == PLAYER_PIECE:
                pygame.draw.circle(screen, color1, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == AI_PIECE:
                pygame.draw.circle(screen, color2, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()


# function to draw text
def draw_text(text,x,y,color,font):
    text_surface=font.render(text,True,color)
    screen.blit(text_surface,(x-text_surface.get_width() // 2,y))

# Celebration function
def celebrate_winner(winner,color,winning_positions):
    # Flash winning line
    for pos in winning_positions:
        r, c = pos
        pygame.draw.circle(screen, CYAN, (
            int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS + 5, 5)
    pygame.display.update()
    pygame.time.wait(2000)
    # Flashing text for celebration
    for _ in range(3):
        screen.fill(BLACK)
        label = myfont.render(f"{winner} wins!!", 1, color)
        screen.blit(label, (width // 2 - label.get_width() // 2, 50))
        pygame.display.update()
        pygame.time.wait(500)
        screen.fill(BLACK)
        pygame.display.update()
        pygame.time.wait(500)

    # Confetti or random particles
    for _ in range(50):
        pygame.draw.circle(screen, random.choice([RED, YELLOW]),(random.randint(0, width), random.randint(0, height // 2)), random.randint(5, 10))
        pygame.display.update()
        pygame.time.wait(30)

    draw_text('Press Space to play again',width//2,500,WHITE,font)
    pygame.display.update()
    exit=False

    while not exit:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                exit=True
                pygame.quit()
                sys.exit()
                
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_SPACE:
                    welcome()
    

# Title animation function
def animate_title():
    screen.fill(BLACK)
    title = "CONNECT4"
    for i in range(len(title) + 1):
        screen.fill(BLACK)
        text = title_font.render(title[:i], True, WHITE)
        screen.blit(text, (width // 2 - text.get_width() // 2, height // 3))
        pygame.display.update()
        pygame.time.wait(200)



two_player=False

def welcome():
    global two_player
    exit_game=False
    screen.fill(DARK_BLUE)
    draw_text("Welcome to Connect Four!", width // 2, 50, WHITE, big_font)
    draw_text("Play With:", width // 2, 120, WHITE, font)

    # ai Button
    ai_button = pygame.Rect(width // 2 - 120, 180, 100, 50)
    pygame.draw.rect(screen, NEON_GREEN, ai_button)

    # Friend Button
    friend_button = pygame.Rect(width // 2 + 20, 180, 100, 50)
    pygame.draw.rect(screen, SOFT_ORANGE, friend_button)

    draw_text("AI", width // 2 - 70, 190, BLACK, font)
    draw_text("Friend", width // 2 + 70, 190, BLACK, font)

    pygame.display.flip()  # Update display

    while not exit_game:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                exit_game=True
            if event.type==pygame.MOUSEBUTTONDOWN:
                if ai_button.collidepoint(event.pos):
                    two_player=False
                    aiPlay()
                if friend_button.collidepoint(event.pos):
                    two_player=True
                    friendPlay()
        pygame.display.update()

def aiPlay():
    global player_color,ai_color
    exit_game=False
    screen.fill(DARK_BLUE)
    draw_text('Playing with AI:',width//2,50,WHITE,big_font)
    draw_text('Choose your color:',width//2, 120,WHITE,font)
    # Red Button
    red_button = pygame.Rect(width // 2 - 120, 180, 100, 50)
    pygame.draw.rect(screen, RED, red_button)

    # Yellow Button
    yellow_button = pygame.Rect(width // 2 + 20, 180, 100, 50)
    pygame.draw.rect(screen, YELLOW, yellow_button)

    draw_text("Red", width // 2 - 70, 190, WHITE, font)
    draw_text("Yellow", width // 2 + 70, 190, BLACK, font)

    pygame.display.flip()  # Update display

    while not exit_game:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                exit_game=True
            if event.type==pygame.MOUSEBUTTONDOWN:
                if red_button.collidepoint(event.pos):
                    player_color=RED
                    ai_color=YELLOW
                    gameLoop()
                if yellow_button.collidepoint(event.pos):
                    player_color=YELLOW
                    ai_color=RED
                    gameLoop()
        pygame.display.update()



def friendPlay():
    global P1_color,P2_color
    exit_game=False
    screen.fill(DARK_BLUE)
    draw_text('Playing with Friend:',width//2,50,WHITE,big_font)
    draw_text('Player 1 choose your color:',width//2, 120,WHITE,font)
    # Red Button
    red_button = pygame.Rect(width // 2 - 120, 180, 100, 50)
    pygame.draw.rect(screen, RED, red_button)

    # Yellow Button
    yellow_button = pygame.Rect(width // 2 + 20, 180, 100, 50)
    pygame.draw.rect(screen, YELLOW, yellow_button)

    draw_text("Red", width // 2 - 70, 190, WHITE, font)
    draw_text("Yellow", width // 2 + 70, 190, BLACK, font)

    pygame.display.flip()  # Update display

    while not exit_game:
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                exit_game=True
            if event.type==pygame.MOUSEBUTTONDOWN:
                if red_button.collidepoint(event.pos):
                    P1_color=RED
                    P2_color=YELLOW
                    gameLoop()
                if yellow_button.collidepoint(event.pos):
                    P1_color=YELLOW
                    P2_color=RED
                    gameLoop()
        pygame.display.update()

def gameLoop():
    board = create_board()
    game_over = False
    
    draw_board(board)
    pygame.display.update()

    myfont = pygame.font.SysFont("monospace", 75)

    turn = random.randint(PLAYER, AI)

    if(not two_player): # play with ai
        while not game_over:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    posx = event.pos[0]
                    if turn == PLAYER:
                        pygame.draw.circle(screen, player_color, (posx, int(SQUARESIZE / 2)), RADIUS)

                pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    
                    # Ask for Player 1 Input
                    if turn == PLAYER:
                        posx = event.pos[0]
                        col = int(math.floor(posx / SQUARESIZE))

                        if is_valid_location(board, col):
                            row = get_next_open_row(board, col)
                            drop_piece(board, row, col, PLAYER_PIECE)

                            win_details=winning_move(board, PLAYER_PIECE)
                            if win_details[0]:
                                draw_board(board)
                                pygame.display.update()
                                celebrate_winner("You",player_color,win_details[1])
                                game_over = True

                            turn += 1
                            turn = turn % 2

                            draw_board(board)

            # # Ask for Player 2 Input
            if turn == AI and not game_over:

                col, minimax_score = minimax(board, 5, -math.inf, math.inf, True)

                if is_valid_location(board, col):
                    # pygame.time.wait(500)
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, AI_PIECE)

                    win_details=winning_move(board, AI_PIECE)
                    if win_details[0]:
                        draw_board(board)
                        pygame.display.update()
                        celebrate_winner("AI",ai_color,win_details[1])
                        game_over = True

                    draw_board(board)

                    turn += 1
                    turn = turn % 2
            if game_over:
                pygame.time.wait(3000)

    else: # play with friend
        while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()

                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    posx = event.pos[0]
                    color = P2_color if turn == 0 else P1_color
                    pygame.draw.circle(screen, color, (posx, int(SQUARESIZE / 2)), RADIUS)
                    pygame.display.update()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    pygame.draw.rect(screen, BLACK, (0, 0, width, SQUARESIZE))
                    posx = event.pos[0]
                    col = int(math.floor(posx / SQUARESIZE))

                    if is_valid_location(board, col):
                        row = get_next_open_row(board, col)
                        piece = 1 if turn == 0 else 2
                        drop_piece(board, row, col, piece)

                        win_details=winning_move(board, piece)
                        if win_details[0]:
                            winner = "Player 1" 
                            color=P1_color
                            if piece == 1:
                                winner="Player 2"
                                color=P2_color

                            # Show the board for 2 seconds before the celebration
                            draw_board(board)
                            pygame.display.update()

                            celebrate_winner(winner,color,win_details[1])
                            game_over = True

                        # Don't redraw the board if game is over
                        if not game_over:
                            draw_board(board)

                        turn += 1
                        turn %= 2
            if game_over:
                pygame.time.wait(3000)  # Wait for 3 seconds before quitting

      
pygame.init()

SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT + 1) * SQUARESIZE

size = (width, height)
RADIUS = int(SQUARESIZE / 2 - 5)

screen = pygame.display.set_mode(size)

#set color for ai game
player_color=RED
ai_color=YELLOW

# set color for friend game
P1_color=RED
P2_color=YELLOW

# initialize font
pygame.font.init()
font = pygame.font.Font(None, 36)
big_font = pygame.font.Font(None, 50)
myfont = pygame.font.SysFont("monospace", 75)
title_font = pygame.font.SysFont("monospace", 120)

# call functions
animate_title()
pygame.time.wait(500)
welcome()