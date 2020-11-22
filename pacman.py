import sys, time
from collections import deque
import heapq
from random import randint, shuffle, choice
import pygame
import tracemalloc

# Game attributes
WORLD_SIZE = 320
BLOCK_SIZE = int(WORLD_SIZE / 20)
LEVEL_NO = 'minimax'
GAME_SPEED = 0.1
RANDOM_GHOST_STEP = 3
DIFFICULTY = 1


class Block(pygame.sprite.Sprite):
    """Wall block."""
    def __init__(self, x: int, y: int):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img16/wall.png").convert_alpha()
        self.rect = self.image.get_rect(center=(x+8, y+8))


class Fruit(pygame.sprite.Sprite):
    """Fruit:)."""
    def __init__(self, x: int, y: int):
        """Find random place on map and inits there."""
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img16/fruit.gif").convert_alpha()
        self.rect = self.image.get_rect(center=(x, y))

    def get_position(self) -> (int, int):
        """Get pacman position."""
        pos_x = self.rect.x // BLOCK_SIZE
        pos_y = self.rect.y // BLOCK_SIZE
        return pos_x, pos_y


class Creature(pygame.sprite.Sprite, ):
    image = rect = None
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

        if keys[pygame.K_LEFT] and WORLD[pos_y][pos_x - 1] != '=':
            self.left()
        elif keys[pygame.K_RIGHT] and WORLD[pos_y][pos_x + 1] != '=':
            self.right()
        elif keys[pygame.K_UP] and WORLD[pos_y - 1][pos_x] != '=':
            self.up()
        elif keys[pygame.K_DOWN] and WORLD[pos_y + 1][pos_x] != '=':
            self.down()

    def right(self):
        self.rect.x += BLOCK_SIZE

    def left(self):
        self.rect.x -= BLOCK_SIZE

    def down(self):
        self.rect.y += BLOCK_SIZE

    def up(self):
        self.rect.y -= BLOCK_SIZE

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

    def manhattan_sort(self, stack: list, obj):
        """Sort route list using manhattan distance.

        From bigger to lower.
        """
        pos_x, pos_y = self.get_position()
        obj_x, obj_y = obj.get_position()

        val_func = lambda x1, x2: (x1 - x2) / abs(x1 - x2) if x1 != x2 else -1
        val_route = {
            'left':  val_func(pos_x, obj_x),
            'right': val_func(obj_x, pos_x),
            'up':    val_func(pos_y, obj_y),
            'down':  val_func(obj_y, pos_y)
        }
        stack.sort(key=lambda x: abs(abs(pos_x - obj_x) + abs(pos_y - obj_y) - val_route[x]), reverse=True)


class Ghost(Creature):
    """Ghost:)."""
    def __init__(self, x:int, y:int):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img16/ghost.gif").convert_alpha()
        self.rect = self.image.get_rect(center=(x, y))
        self.random_step = randint(0, DIFFICULTY * 3)

    def manhattan_step(self, pacman, ghosts_positions: set):
        pos_x, pos_y = self.get_position()
        route = self.get_route()
        for position in ghosts_positions:
            if pos_x - 1 == position[0] and 'left' in route:
                route.remove('left')
            elif pos_x + 1 == position[0] and 'right' in route:
                route.remove('right')
            elif pos_y + 1 == position[1] and 'down' in route:
                route.remove('down')
            elif pos_y - 1 == position[1] and 'up' in route:
                route.remove('up')

        self.manhattan_sort(route, pacman)
        if route:
            if self.random_step < DIFFICULTY * 5:
                self.func_way(route[-1])
                self.random_step += 1
            else:
                self.func_way(choice(route))
                self.random_step = randint(0, DIFFICULTY * 3)


class Pacman(Creature):
    """Pacman character."""
    hunger = 0

    opposite_way = {
        'left': 'right',
        'right': 'left',
        'down': 'up',
        'up': 'down',
    }

    def __init__(self, x: int, y: int):
        """Find random place for pacman and inits there."""
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load("img16/pacman-right.gif").convert_alpha()
        self.rect = self.image.get_rect(center=(x, y))

    def right(self):
        self.image = pygame.image.load("img16/pacman-right.gif").convert_alpha()
        self.rect.x += BLOCK_SIZE

    def left(self):
        self.image = pygame.image.load("img16/pacman-left.gif").convert_alpha()
        self.rect.x -= BLOCK_SIZE

    def down(self):
        self.image = pygame.image.load("img16/pacman-down.gif").convert_alpha()
        self.rect.y += BLOCK_SIZE

    def up(self):
        self.image = pygame.image.load("img16/pacman-up.gif").convert_alpha()
        self.rect.y -= BLOCK_SIZE

    def finished(self):
        """Check if pacman found a fruit."""
        pos_x, pos_y = self.get_position()
        if WORLD[pos_y][pos_x] == '*':
            return True

    def dfs(self, greedy=False, fruit=None):
        """Finds a way to the fruit using depth-first search algorithm."""
        now = time.time()
        steps = 0
        stack_route = [self.get_route()]
        way_back = []

        while stack_route:
            time.sleep(GAME_SPEED)
            if stack_route[-1]:
                if greedy:
                    self.manhattan_sort(stack_route[-1], fruit)
                nxt = stack_route[-1].pop()
                self.func_way(nxt)
                stack_route.append(self.get_route())
                back = self.opposite_way[nxt]
                if back in stack_route[-1]:
                    stack_route[-1].remove(back)
                way_back.append(back)
            else:
                stack_route.pop()
                nxt = way_back.pop()
                self.func_way(nxt)
            steps += 1
            draw()
            if self.finished():
                break

        if not greedy:
            print(f"Statistic for DFS:\n\tSteps: {steps}")
        else:
            print(f"Statistic for greedy algorithm:\n\tSteps: {steps}")
        print(f"\tTime: {round(time.time() - now, 3)}")

    def bfs(self, a_star=False, fruit=None):
        """Finds a way to the fruit using breadth-first search algorithm.

        if a_star = True => uses A* algorithm
        """
        now = time.time()
        steps = 0
        if a_star:
            queue = []
            for route in self.get_route():
                heapq.heappush(queue, (1, [route]))
        else:
            queue = deque()
            queue.extend([[route] for route in self.get_route()])
        current_way = []
        was_here = set()

        def make_step(step):
            time.sleep(GAME_SPEED)
            self.func_way(step)
            draw()
            nonlocal steps
            steps += 1

        while queue:
            if a_star:
                nxt = heapq.heappop(queue)[1]
            else:
                nxt = queue.popleft()
            while len(current_way) > len(nxt):
                make_step(self.opposite_way[current_way.pop()])
            while nxt[:len(current_way)] != current_way:
                make_step(self.opposite_way[current_way.pop()])
            for n in nxt[len(current_way):]:
                current_way.append(n)
                make_step(n)
            if not (place := self.get_position()) in was_here:
                if self.finished():
                    print('found')
                    break
                was_here.add(place)
                for way in self.get_route():
                    if way != self.opposite_way[current_way[-1]]:
                        if a_star:
                            heapq.heappush(queue, (self.a_star_heuristic(list(current_way + [way]), fruit), list(current_way + [way])))
                        else:
                            queue.append(list(current_way + [way]))

        if a_star:
            print(f"Statistic for A*:\n\tSteps: {steps}")
        else:
            print(f"Statistic for BFS:\n\tSteps: {steps}")
        print(f"\tTime: {round(time.time() - now, 3)}")

    def a_star_heuristic(self, element: list, fruit: Fruit):
        """Return heuristic for current position in A* algorithm."""
        pos_x, pos_y = self.get_position()
        fruit_x = fruit.rect.x // BLOCK_SIZE
        fruit_y = fruit.rect.y // BLOCK_SIZE

        return abs(pos_x - fruit_x) + abs(pos_y + fruit_y) + len(element)

    @staticmethod
    def expect(way, ghost):
        x, y = ghost.get_position()
        if way == 'left':
            x -= 1
        elif way == 'right':
            x += 1
        elif way == 'down':
            y += 1
        elif way == 'up':
            y -= 1

        return x, y

    def min_sort(self, way, ghost):
        x, y = self.expect(way, ghost)
        pos_x, pos_y = self.get_position()
        return abs(pos_x - x) + abs(pos_y - y)

    def max_sort(self, way, g_conditions, f_conditions):
        exp_x, exp_y = self.expect(way, self)
        pos_x, pos_y = self.get_position()
        heuristic = 0
        for x0, y0 in g_conditions:
            distance = abs(exp_x - x0) + abs(exp_y - y0)
            gradient = abs(exp_x + pos_x - 2 * x0) + abs(exp_y + pos_y - 2 * y0)
            heuristic += (gradient / (distance + 1)) * 5

        for x0, y0 in f_conditions:
            distance = abs(exp_x - x0) + abs(exp_y - y0)
            gradient = abs(exp_x + pos_x - 2 * x0) + abs(exp_y + pos_y - 2 * y0)
            heuristic += (distance / gradient) / (1 + (self.hunger / 100q))

        return heuristic


    def minimax_step(self, ghosts, fruits):
        pos_x, pos_y = self.get_position()
        ghosts_filtered = []
        for ghost in ghosts:
            g_x, g_y = ghost.get_position()
            if abs(pos_x - g_x) + abs(pos_y - g_y) <= 12:
                ghosts_filtered.append(ghost)

        ghost_conditions = []
        for ghost in ghosts_filtered:
            g_route = ghost.get_route()
            g_route = sorted(g_route, key=lambda r: self.min_sort(r, ghost))
            expected_step = g_route[0]
            expected_location = self.expect(expected_step, ghost)
            ghost_conditions.append(expected_location)

        fruit_conditions = []
        for fruit in fruits:
            fruit_conditions.append(fruit.get_position())

        route = self.get_route()
        route.sort(key = lambda r: self.max_sort(r, ghost_conditions, fruit_conditions))
        self.func_way(route[0])
        self.hunger += 1
        for fruit in fruits:
            if self.get_position() == fruit.get_position():
                fruits.remove(fruit)
                self.hunger = 0


def main():
    # Set title
    pygame.display.set_caption('Pacman')

    # Load level
    load_level(LEVEL_NO)

    # Create walls
    global walls, DIFFICULTY
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

    draw()

    # if launched with arg "play"
    if len(sys.argv) > 1 and sys.argv[1] == 'play':
        # Start game loop
        while True:
            # Listen events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    keys = pygame.key.get_pressed()
                    pac.update(keys)
                pygame.display.flip()
    if len(sys.argv) > 1 and sys.argv[1] == 'minimax':
        if len(sys.argv) > 2:
            DIFFICULTY = int(sys.argv[2])
        fruits = [fruit]
        for _ in range(DIFFICULTY * 5):
            x_fruit, y_fruit = get_random_place('*')
            fruits.append(Fruit(x_fruit, y_fruit))

        ghosts = []
        if DIFFICULTY > 2:
            DIFFICULTY //= 2
            DIFFICULTY += 1
        for _ in range(DIFFICULTY):
            x_ghost, y_ghost = get_random_place('$')
            ghosts.append(Ghost(x_ghost, y_ghost))

        drawer = all_draw(fruits, ghosts, pac)
        drawer()
        time.sleep(3)
        caught = False
        while len(fruits):
            time.sleep(GAME_SPEED)

            ghosts_positions = set()
            for ghost in ghosts:
                ghosts_positions.add(ghost.get_position())

            if pac.get_position() in ghosts_positions:
                caught = True

            if not caught:
                for ghost in ghosts:
                    ghost.manhattan_step(pac, ghosts_positions)
                    if ghost.get_position() == pac.get_position():
                        caught = True

            if caught:
                print('GAME OVER!')
                pac.rect = (-100, -100)
                drawer(end=True)
                break
            pac.minimax_step(ghosts, fruits)
            drawer()
        else:
            print('PACMAN WIN!')

    else:
        # DFS
        run_alg(pac.dfs)
        #
        # Replace pacman to starting (random) position
        pac = Pacman(x_pac, y_pac)

        # A*
        run_alg(pac.bfs, arg={"a_star": True, "fruit": fruit})

        # Replace pacman to starting (random) position
        pac = Pacman(x_pac, y_pac)

        # Greedy algorithm
        run_alg(pac.dfs, arg={"greedy": True, "fruit": fruit})

        # Replace pacman to starting (random) position
        pac = Pacman(x_pac, y_pac)

        # BFS
        run_alg(pac.bfs)

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

def all_draw(fruits, ghosts, pacman):
    def anon_draw(end=False):
        screen.fill((0, 0, 0))
        walls.draw(screen)
        if not end:
            screen.blit(pacman.image, pacman.rect)
        for fruit in fruits:
            screen.blit(fruit.image, fruit.rect)
        for ghost in ghosts:
            screen.blit(ghost.image, ghost.rect)
        pygame.display.update()
    return anon_draw


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
    return x + 8, y + 8


def run_alg(func, arg={}):
    """Run algorithm, measure memory costs."""
    time.sleep(2)
    draw()
    time.sleep(2)
    tracemalloc.start()
    func(**arg) if arg else func()
    snapshot = tracemalloc.take_snapshot()
    stats = snapshot.statistics('lineno')
    total = sum([s.size for s in stats])
    print(f"\tTotal memory usage: {total/10**3}MB")
    tracemalloc.stop()


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WORLD_SIZE, WORLD_SIZE))
    main()
