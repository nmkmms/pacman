import sys, time
from collections import deque
from random import randint, shuffle
import pygame
import tracemalloc

# Game attributes
BLOCK_SIZE = 32
LEVEL_NO = 2


class Block(pygame.sprite.Sprite):
    """Wall block."""
    def __init__(self, x: int, y: int):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img/wall.png").convert_alpha()
        self.rect = self.image.get_rect(center=(x+16, y+16))


class Fruit(pygame.sprite.Sprite):
    """Fruit:)."""
    def __init__(self, x: int, y: int):
        """Find random place on map and inits there."""
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img/fruit.gif").convert_alpha()

        self.rect = self.image.get_rect(center=(x, y))


class Pacman(pygame.sprite.Sprite):
    """Pacman character."""

    opposite_way = {
        'left': 'right',
        'right': 'left',
        'down': 'up',
        'up': 'down',
    }


    def __init__(self, x: int, y: int):
        """Find random place for pacman and inits there."""
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img/pacman-right.gif").convert_alpha()

        self.rect = self.image.get_rect(center=(x, y))


    def get_position(self) -> (int, int):
        """Get pacman position."""
        pos_x = self.rect.x // BLOCK_SIZE
        pos_y = self.rect.y // BLOCK_SIZE
        return pos_x, pos_y


    def update(self, keys):
        """Moves pacman using keys input.

        Change image according to pacman's moves.
        """
        pos_x, pos_y = self.get_position()

        if keys[pygame.K_LEFT] and not WORLD[pos_y][pos_x - 1] == '=':
            self.left()
        elif keys[pygame.K_RIGHT] and not WORLD[pos_y][pos_x + 1] == '=':
            self.right()
        elif keys[pygame.K_UP] and not WORLD[pos_y - 1][pos_x] == '=':
            self.up()
        elif keys[pygame.K_DOWN] and not WORLD[pos_y + 1][pos_x] == '=':
            self.down()


    def right(self):
        self.image = pygame.image.load("img/pacman-right.gif").convert_alpha()
        self.rect.x += BLOCK_SIZE


    def left(self):
        self.image = pygame.image.load("img/pacman-left.gif").convert_alpha()
        self.rect.x -= BLOCK_SIZE


    def down(self):
        self.image = pygame.image.load("img/pacman-down.gif").convert_alpha()
        self.rect.y += BLOCK_SIZE


    def up(self):
        self.image = pygame.image.load("img/pacman-up.gif").convert_alpha()
        self.rect.y -= BLOCK_SIZE


    def finished(self):
        """Check if pacman found a fruit."""
        pos_x, pos_y = self.get_position()
        if WORLD[pos_y][pos_x] == '*':
            return True

    def get_route(self):
        """Adds possible routes from current location."""
        pos_x, pos_y = self.get_position()

        leafs = []
        if not WORLD[pos_y][pos_x + 1] == '=':
            leafs.append('right')
        if not WORLD[pos_y][pos_x - 1] == '=':
            leafs.append('left')
        if not WORLD[pos_y + 1][pos_x] == '=':
            leafs.append('down')
        if not WORLD[pos_y - 1][pos_x] == '=':
            leafs.append('up')
        shuffle(leafs)
        return leafs

    def func_way(self, way: str):
        eval(f"self.{way}()")


    def dfs(self):
        """Finds a way to the fruit using breadth-first search algorithm."""
        steps = 0
        stack_route = [self.get_route()]
        way_back = []

        while stack_route[-1] != 'found':
            time.sleep(0.01)
            if stack_route[-1]:
                next = stack_route[-1].pop()
                self.func_way(next)
                stack_route.append(self.get_route())
                back = self.opposite_way[next]
                if back in stack_route[-1]:
                    stack_route[-1].remove(back)
                way_back.append(back)
            else:
                stack_route.pop()
                next = way_back.pop()
                self.func_way(next)
            steps += 1
            draw()
            if self.finished():
                break

        print(f"Statistic for DFS:\n\tSteps: {steps}")

    
    def manhattan_sort(self, stack: list, fruit: Fruit):
        """Sort route list using manhattan distance.
        
        From bigger to lower.
        """
        pos_x, pos_y = self.get_position()
        fruit_x = fruit.rect.x // BLOCK_SIZE
        fruit_y = fruit.rect.y // BLOCK_SIZE

        route = {'left': -1, 'right': 1, 'up': -1, 'down': 1}
        stack.sort(key=lambda x: abs(pos_x - fruit_x + pos_y - fruit_y + route[x]), reverse=True)


    def greedy(self, fruit: Fruit):
        """Finds a way to the fruit using greedy algorithm."""
        steps = 0
        stack_route = [self.get_route()]
        way_back = []

        while stack_route[-1] != 'found':
            time.sleep(0.01)
            if stack_route[-1]:
                self.manhattan_sort(stack_route[-1], fruit)
                next = stack_route[-1].pop()
                self.func_way(next)
                stack_route.append(self.get_route())
                back = self.opposite_way[next]
                if back in stack_route[-1]:
                    stack_route[-1].remove(back)
                way_back.append(back)
            else:
                stack_route.pop()
                next = way_back.pop()
                self.func_way(next)
            steps += 1
            draw()
            if self.finished():
                break

        print(f"Statistic for greedy algorithm:\n\tSteps: {steps}")




def main():
    # Set title
    pygame.display.set_caption('Pacman')

    # Load level
    load_level(LEVEL_NO)

    # Create walls
    global walls
    walls = pygame.sprite.Group()
    for y, row in enumerate(WORLD):
        for x, block in enumerate(row):
            if block == '=':
                block = Block(x * BLOCK_SIZE, y * BLOCK_SIZE)
                walls.add(block)

    # Create pacman and fruit
    global pac, fruit
    # Chose random place for pacman
    x_pac, y_pac = get_random_place('@')
    pac = Pacman(x_pac, y_pac)
    # Chose random place for fruit
    x_fruit, y_fruit = get_random_place('*')
    fruit = Fruit(x_fruit, y_fruit)

    # if launched with arg "play"
    if len(sys.argv) > 1 and sys.argv[1] == 'play':
        # Start game loop
        while True:
            # Redraw background and pacman
            draw()

            # Listen events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    pac.update(keys)
                pygame.display.flip()
    else:
        draw()

        # DFS
        run_alg(pac.dfs)

        # Replace pacman to starting (random) position
        pac = Pacman(x_pac, y_pac)

        # Greedy alogithm
        run_alg(pac.greedy, arg=fruit)

        # Infinite loop, waiting for closing
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()


def draw():
    screen.fill((0, 0, 0))
    walls.draw(screen)
    screen.blit(pac.image, pac.rect)
    screen.blit(fruit.image, fruit.rect)
    pygame.display.update()


def load_level(number: int):
    """Load level from .txt file from levels directory."""
    global WORLD
    WORLD = []
    file = f"levels/level-{number}.txt"
    with open(file) as in_file:
        for line in in_file:
            row = []
            for block in line.strip():
                row.append(block)
            WORLD.append(row)


def get_random_place(character: str):
    """Return random (x, y) position for pacman or fruit."""
    x = y = 0
    while not WORLD[y][x] == ' ':
        x = randint(1, 18)
        y = randint(1, 18)

    # Add to level list
    WORLD[y][x] = character

    # Scale
    x *= BLOCK_SIZE
    y *= BLOCK_SIZE
    return (x + 16, y + 16)


def run_alg(func, arg=None):
    """Run algorithm, measure memory costs."""
    time.sleep(2)
    tracemalloc.start()
    func(arg) if arg else func()
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('lineno')
    total = sum([s.size for s in stats])
    print(f"\tTotal memory usage: {total/10**3}MB")
    tracemalloc.stop()


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((640, 640))
    main()
