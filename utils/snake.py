class Snake:

    def __init__(self, start_pos, start_len=5, start_dir=1):
        self.head_pos = start_pos   # 2 element list: [x, y]
        self.direction = start_dir  # integer from 0-3 inclusive; 0 is up, going around clockwise
        self.alive = True
        self.score = 0

        self.scales_pos = []

        # grow the snake depending on starting length
        self.length = 0
        for i in range(start_len):
            self.grow()

    def move(self):

        # move each scale to the position ahead
        for i in range(0, self.length):
            self.scales_pos[self.length-i-1] = self.scales_pos[self.length-i-2]
            self.scales_pos[0] = self.head_pos

        # move head
        direction_moves = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        move = direction_moves[self.direction]
        self.head_pos = [self.head_pos[0] + move[0], self.head_pos[1] + move[1]]

    def change_direction(self, dir_change):
        # dir_change should be either 1 or -1; 1 is right, -1 is left
        self.direction = (self.direction + dir_change) % 4

    def grow(self):
        self.scales_pos.append([-1, -1])
        self.length += 1

    def get_positions(self):
        return [self.head_pos] + self.scales_pos

