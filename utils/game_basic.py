
from utils.snake import *
import pygame
from time import sleep
from random import randint

# load images
file_path = 'C:/Users/Nathan Kang (BCP)/OneDrive - Bellarmine College Preparatory/' \
            '80 Programming Projects/AISnakes/utils/images/'
print(file_path + '/background.png')
background = pygame.image.load(file_path + 'background.png')         # size: 1000, 1000
green_square = pygame.image.load(file_path + 'green_square.png')     # size: 20, 20
red_square = pygame.image.load(file_path + 'red_square.png')         # size: 20, 20
blue_square = pygame.image.load(file_path + 'blue_square.png')       # size: 20, 20
end_screen = pygame.image.load(file_path + 'end_screen.png')         # size: 820, 400
orange_line = pygame.image.load(file_path + 'orange_line.png')       # size: 1000, 10
logo = pygame.image.load(file_path + 'logo32x32.png')                # size: 32, 32


class Game:

    def __init__(self, board_size=None, frame_time=0.1, starting_length=5, num_apples=1, show_game=True,
                 num_snakes=2, frame_worth=0, apple_worth=1, max_frames=10000):
        """
        Creates a new Game object which displays a BattleSnake game and controls the play of the game.
        - only works with 1 or 2 snakes
        - apples are randomly initiated

        :param board_size: a tuple with 2 values: (width, height), defaults to (45, 25)
        :param frame_time: a float which controls the delay in seconds between frames
        :param starting_length: the starting length of each snake
        :param num_apples: the number of apples in the game
        :param show_game: a boolean which controls whether to show the gui of the game
        :param num_snakes: an int with the number of snakes, either 1 or 2
        :param frame_worth: an int which controls the score bonus for every frame survived
        :param apple_worth: an int which controls the score bonus for every apple eaten
        """

        # init board size and frame time
        if board_size is None:
            self.board_size = (44, 25)
        else:
            self.board_size = board_size
        self.frame_time = frame_time
        self.max_frames = max_frames

        Snake([10, 13], start_len=starting_length, start_dir=1)

        # init snakes
        if num_snakes == 1:
            self.snakes = [Snake([10, 13], start_len=starting_length, start_dir=1)]
        else:
            self.snakes = []
            # first snake on the right, facing left
            self.snakes.append(Snake([10, 10], start_len=starting_length, start_dir=1))
            # second snake on the left, facing right
            self.snakes.append(Snake([34, 15], start_len=starting_length, start_dir=-1))

        # init apple
        self.apples = []
        for _ in range(num_apples):
            self.apples.append(self.gen_apple())

        # init scoring system
        self.frame_worth = frame_worth
        self.apple_worth = apple_worth

        # init pygame
        pygame.init()
        self.running = True

        self.show_game = show_game
        if self.show_game:
            # set the logo and name
            pygame.display.set_icon(logo)
            pygame.display.set_caption("Battle Snake")

            # create a surface on screen
            screen_size = (self.board_size[0] * 20, (self.board_size[1] + 2) * 20)
            self.screen = pygame.display.set_mode((screen_size[0], screen_size[1]))

    def run_game(self):

        num_frames = 0

        # game loop
        while self.running:

            num_frames += 1

            if self.show_game:
                self.update_screen()
                sleep(self.frame_time)

            # event handling
            self.decision_handle()

            # check deaths
            self.check_deaths()

            # move snakes if alive and check apples and check if dead

            both_dead = self.move_and_score()

            if both_dead or num_frames > self.max_frames:
                self.running = False

    def move_and_score(self):
        dead = True
        for s in self.snakes:
            if s.alive:
                s.move()
                s.score += self.frame_worth
            if s.head_pos in self.apples and s.alive:
                s.grow()
                s.score += self.apple_worth
                self.reset_apple(s.head_pos)
            if s.alive:
                dead = False
        return dead

    def gen_apple(self):
        """
        Generates a new position for an apple which is inside the board and not overlapping a snake.

        :return: a position in the form of a list: [x, y]
        """

        new_pos = [randint(1, self.board_size[0]), randint(1, self.board_size[1])]
        while new_pos in self.get_snake_positions():
            new_pos = [randint(1, self.board_size[0]), randint(1, self.board_size[1])]

        return new_pos

    def get_snake_positions(self):
        positions = []
        for s in self.snakes:
            positions += s.get_positions()
        return positions

    def find_screen_pos(self, pos):
        x_pos = (pos[0] - 1) * 20
        y_pos = (pos[1] - 1) * 20
        screen_pos = (x_pos, y_pos)
        return screen_pos

    def blit_text(self, text, color, pos):
        font = pygame.font.Font('freesansbold.ttf', 24)
        text = font.render(text, False, color)
        text_rect = text.get_rect()
        text_rect.center = pos
        self.screen.blit(text, text_rect)

    def in_board(self, pos):
        return not (pos[0] < 1 or pos[0] > self.board_size[0]
                    or pos[1] < 1 or pos[1] > self.board_size[1])

    def reset_apple(self, apple_pos):
        self.apples.remove(apple_pos)
        self.apples.append(self.gen_apple())

    def check_deaths(self):
        for s in self.snakes:
            snake_positions = self.get_snake_positions()
            snake_positions.remove(s.head_pos)
            if s.head_pos in snake_positions:
                s.alive = False
            elif not self.in_board(s.head_pos):
                s.alive = False

    def update_screen(self):
        # update background
        self.screen.blit(background, (0, 0))

        # update apple
        for a in self.apples:
            self.screen.blit(red_square, self.find_screen_pos(a))

        # show snakes
        for i in range(len(self.snakes)):
            for pos in self.snakes[i].get_positions():
                if i == 0:
                    self.screen.blit(green_square, self.find_screen_pos(pos))
                else:
                    self.screen.blit(blue_square, self.find_screen_pos(pos))

        # show info bar
        self.screen.blit(orange_line, self.find_screen_pos((0, self.board_size[1]+1)))

        info = ''
        for i in range(len(self.snakes)):
            info += f'Player {i+1} : {self.snakes[i].score}             '
        self.blit_text(info, (249, 234, 31), (440, 520))

        # update screen
        pygame.display.flip()

    def decision_handle(self):
        for event in pygame.event.get():
            # check keystroke and change velocity
            if event.type == pygame.KEYDOWN:
                unicode = event.dict['unicode']
                key = event.dict['key']

                # update player 1
                if unicode == "a":
                    self.snakes[0].change_direction(-1)
                elif unicode == "d":
                    self.snakes[0].change_direction(1)

                # update player 2
                if key == pygame.K_LEFT:
                    self.snakes[-1].change_direction(-1)
                if key == pygame.K_RIGHT:
                    self.snakes[-1].change_direction(1)

            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                self.running = False
