from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from math import floor, ceil, sqrt
from random import uniform
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


def make_normal_dist_array_0_1(N, sigma):
    lower, upper = 0, 1
    mu = 0.5
    samples = scipy.stats.truncnorm.rvs(
        (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
    return samples


def round_float(f):
    if ceil(f) - f < f - floor(f):
        return ceil(f)
    return floor(f)


class Student(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, mod, network, voice, talk_prob):  # schedule
        super().__init__(unique_id, mod)
        self.talk_prob = talk_prob
        self.nw = network
        self.voice = voice
        self.covid_measures = False
        self.fancy_classrooms = False
        self.destination = None
        self.id = unique_id
        # self.desired_n_friends = traits[0]      # "desired number of friends"
        # self.xtra_version = traits[1]           # "how extraverted: float in (0, 1)"
        # self.loneliness_tolerance = traits[2]   # "loneliness tolerance: float in (0, 1)"

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, radius=1, include_center=False
        )
        truly_possible_steps = []
        for step in possible_steps:
            if not self.model.grid.get_cell_list_contents([step]):
                truly_possible_steps.append(step)
        if len(truly_possible_steps) > 0:
            new_position = self.random.choice(truly_possible_steps)

            # print(self.pos, new_position)
            self.model.grid.move_agent(self, new_position)

    def covid_measure_switch(self):
        self.covid_measures = not self.covid_measures

    def engage_conv(self):
        if self.talk_prob >= uniform(0, 1):
            cellmates = self.model.grid.get_cell_list_contents([self.pos]) if not self.covid_measures \
                else []
            neighbours = []
            for neighbor_cell in self.model.grid.get_neighborhood(
                    self.pos, moore=False, radius=1, include_center=not self.covid_measures):
                neighbours += self.model.grid.get_cell_list_contents([neighbor_cell])
            if cellmates and len(cellmates) > 1:
                other_agent = self.random.choice(cellmates)
                self.nw.step(self.unique_id, other_agent.unique_id)
            elif neighbours and self.voice >= uniform(0, 1):
                other_agent = self.random.choice(cellmates)
                self.nw.step(self.unique_id, other_agent.unique_id)

    def step(self):
        if not self.fancy_classrooms:
            self.move()
            self.engage_conv()


class SchoolModel(Model):
    """A model with some number of agents."""

    def __init__(self, n, network, voice_sigma, talk_prob_sigma):
        self.num_agents = n
        self.nw = network
        self.fancy_classrooms = False
        self.grid = MultiGrid(*self.find_wh(), True)
        self.schedule = RandomActivation(self)
        self.make_agents(voice_sigma, talk_prob_sigma)

    #  def determine_classes(self, avg, tol):

    def make_agents(self, v_sig, tp_sig):
        loudness_of_voices = make_normal_dist_array_0_1(self.num_agents, v_sig)
        talk_probabilities = make_normal_dist_array_0_1(self.num_agents, tp_sig)
        for i, (voice, talk_prob) in enumerate(zip(loudness_of_voices, talk_probabilities)):
            a = Student(i, self, self.nw, voice, talk_prob)  # node zero
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

    def find_wh(self):
        if not self.fancy_classrooms:
            w = ceil(3 * sqrt(self.num_agents))
            return w, w

    def step(self):
        self.schedule.step()


# for i in range(20):
#    model.step()

# print(model.grid.grid)
'''
agent_counts = np.zeros((model.grid.width, model.grid.height))
for cell in model.grid.coord_iter():
    cell_content, x, y = cell
    agent_count = len(cell_content)
    agent_counts[x][y] = agent_count
plt.imshow(agent_counts, interpolation="nearest")
plt.colorbar()
# If running from a text editor or IDE, remember you'll need the following:
plt.show()
'''
