from ABM_classes import SchoolModel, Student
from fast_model import *
import pygame, sys
from pygame.locals import *

""" VISUALIZATION OF THE ABM 
    
    Visualization is done using Pygame. If not installed, type "pip install Pygame" in the terminal
    and press enter.
    
"""


def draw_student(col, x, y, hcw, screen):
    radius = 2*hcw//3
    pygame.draw.circle(screen, col, ((1+2*x)*hcw, (1+2*y)*hcw), radius, (radius//6)+1)


def draw_students(grid, col, hcw, screen):
    ind_x = 0
    ind_y = 0
    for cel in grid:
        n_ag = len(cel)
        if n_ag == 1:
            draw_student(col, ind_x, ind_y, hcw, screen)
        elif n_ag > 1:
            draw_student(Color("blue"), ind_x, ind_y, hcw, screen)
        ind_x += 1
        if ind_x == grid.width:
            ind_x = 0
            ind_y += 1


def draw_grid(grid, px_h, screen):
    screen.fill(Color("white"))
    hcw = px_h//(2*grid.height)

    ind_x = 0
    for col in grid.grid:
        x = (1 + 2*ind_x)*hcw
        ind_y = 0
        for cel in col:
            y = (1+2*ind_y)*hcw
            color = Color("gray") if ind_y%2 == ind_x%2 else Color("white")
            pygame.draw.rect(screen, color, (x-hcw, y-hcw, 2*hcw, 2*hcw), 0)
            ind_y += 1
        ind_x += 1


sq_lst_ind_0 = []


def animate_model(my_model, px_h, max_it):
    pygame.init()
    size = (my_model.grid.width*px_h//my_model.grid.height, px_h)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("ABM Model Animation")
    screen.fill(Color("white"))

    FPS = 20
    clock = pygame.time.Clock()

    it = 0
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == QUIT:
                done = True

        my_model.step()
        it += 1
        if it == max_it:
            done = True

        """ DRAW: """

        draw_grid(my_model.grid, 500, screen)

        hcw = px_h//(2*my_model.grid.height)
        draw_students(my_model.grid, Color("red"), hcw, screen)

        pygame.display.update()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

