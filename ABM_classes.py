from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from math import floor, ceil, sqrt
import numpy as np
import matplotlib.pyplot as plt

def round_float(f):
    if ceil(f)-f < f-floor(f):
        return ceil(f)
    return floor(f)


class Student(Agent):
    """An agent with fixed initial wealth."""

    def __init__(self, unique_id, mod, traits, network): # schedule
        super().__init__(unique_id, mod)
        self.nw = network
        self.covid_measures = False
        self.fancy_classrooms = False
        self.destination = None
        self.id = unique_id
        # self.node_id = node.get_id ?
        # self.grade = GET GRADE FROM DATASET VIA NODE?
        # self.age = CAN WE GET THE AGE FROM THE NODE?
        self.desired_n_friends = traits[0]      # "desired number of friends"
        self.xtra_version = traits[1]           # "how extraverted: float in (0, 1)"
        self.loneliness_tolerance = traits[2]   # "loneliness tolerance: float in (0, 1)"

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
        cellmates = []
        for neighbor_cell in self.model.grid.get_neighborhood(
            self.pos, moore=False, radius=1, include_center=not self.covid_measures):
                cellmates += self.model.grid.get_cell_list_contents([neighbor_cell])
        if len(cellmates) > 1:
            other_agent = self.random.choice(cellmates)
            self.nw.step(self.unique_id, other_agent.unique_id)

    def step(self):
        if not self.fancy_classrooms:
            self.move()
            self.engage_conv()


class SchoolModel(Model):
    """A model with some number of agents."""
    def __init__(self, n, network):
        self.num_agents = n
        self.fancy_classrooms = False
        self.grid = MultiGrid(*self.find_wh(), True)
        self.schedule = RandomActivation(self)
        # Create agents
        for i in range(self.num_agents):
            a = Student(i, self, [4, np.random.normal(.5, .5), np.random.normal(.5, .5)],
                        network) # node zero
            self.schedule.add(a)
            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, (x, y))

  #  def determine_classes(self, avg, tol):

    def find_wh(self):
        if not self.fancy_classrooms:
            w = ceil(2*sqrt(self.num_agents))
            return w, w

    def step(self):
        self.schedule.step()


#for i in range(20):
#    model.step()

#print(model.grid.grid)
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
