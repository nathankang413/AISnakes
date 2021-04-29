from utils.game_basic import Game
from utils.genetic_algo import *

import pygame


class TrainAI(Game):

    def __init__(self, board_size=None, frame_time=0.05, starting_length=10, num_apples=1, show_game=True,
                 num_snakes=1, frame_worth=1, apple_worth=50, max_frames=10000, snake_hunger=1000, move_score=5,
                 brains=None, apple_food=25, away_punishment=1.5):
        super().__init__(board_size=board_size,
                         frame_time=frame_time,
                         starting_length=starting_length,
                         num_apples=num_apples,
                         show_game=show_game,
                         num_snakes=num_snakes,
                         frame_worth=frame_worth,
                         apple_worth=apple_worth,
                         max_frames=max_frames)

        self.max_frames = max_frames
        self.move_score = move_score
        self.away_punishment = away_punishment
        self.apple_food = apple_food

        # init snakes
        for i in range(len(self.snakes)):
            self.snakes[i].brain = brains[i]
            self.snakes[i].hunger = snake_hunger

    def move_and_score(self):
        dead = True
        for s in self.snakes:
            if s.alive:
                prev_pos = s.head_pos
                s.move()
                new_pos = s.head_pos
                s.score += self.frame_worth
                s.score += int(self.closer_to_apple(prev_pos, new_pos) * self.move_score)
                s.hunger -= 1
                if s.hunger <= 0:
                    s.alive = False
            if s.head_pos in self.apples and s.alive:
                s.grow()
                s.score += self.apple_worth
                s.hunger += self.apple_food
                self.reset_apple(s.head_pos)
            if s.alive:
                dead = False
        return dead

    def closer_to_apple(self, prev_pos, new_pos):
        if len(self.apples) == 1:
            apple = self.apples[0]

            prev_dist = math.sqrt((prev_pos[0] - apple[0]) ** 2 + (prev_pos[1] - apple[1]) ** 2)
            new_dist = math.sqrt((new_pos[0] - apple[0]) ** 2 + (new_pos[1] - apple[1]) ** 2)
            if new_dist < prev_dist:
                return prev_dist - new_dist

            return self.away_punishment * (prev_dist - new_dist)
        else:
            return 0

    def decision_handle(self):
        for event in pygame.event.get():
            # only do something if the event is of type QUIT
            if event.type == pygame.QUIT:
                # change the value to False, to exit the main loop
                self.running = False

        for i in range(len(self.snakes)):
            preds = self.snakes[i].brain.predict(self.get_data(i)).tolist()
            decision = preds.index(max(preds)) - 1
            self.snakes[i].change_direction(decision)

    def get_data(self, snake_ind):
        '''
        --- data format --- (18 total values)
        0 - vel_x
        1 - vel_y
        2 - head x
        3 - head y

        4/5 - forward
        6/7 - for-left
        8/9 - left
        10/11 - back-left
        12/13 - for-right
        14/15 - right
        16/17 - back-right

        '''

        snake = self.snakes[snake_ind]
        data = [0] * 18

        # velocity and head pos values
        direction_moves = [[0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]]
        velocities = direction_moves[snake.direction * 2]
        data[0] = velocities[0]
        data[1] = velocities[1]
        data[2] = snake.head_pos[0]
        data[3] = snake.head_pos[1]

        # next 6 data fills: objects in each direction
        directions = [(snake.direction * 2 + i) % 8 for i in [0, -1, -2, -3, 1, 2, 3]]
        to_check = [direction_moves[i] for i in directions]

        data_pos = 4
        for vels in to_check:
            current_pos = [snake.head_pos[0] + vels[0], snake.head_pos[1] + vels[1]]
            counter = 1

            while self.is_object(current_pos) == 0:
                current_pos = [current_pos[0] + vels[0], current_pos[1] + vels[1]]
                counter += 1

            data[data_pos] = counter
            data[data_pos + 1] = self.is_object(current_pos)
            data_pos += 2

        for i in [2, 3, 4, 6, 8, 10, 12, 14, 16]:
            data[i] = data[i] / 50

        return data

    def is_object(self, pos):
        if not self.in_board(pos):
            return -2
        if pos in self.get_snake_positions():
            return -1
        if pos in self.apples:
            return 1
        return 0
