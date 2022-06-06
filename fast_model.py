import numpy as np
import math
import pickle
from multiprocessing import Pool
#from run_combined_model import settings_lst, constant_lst
constant_lst = [5,     # MAX_FRIEND = 5  --> given by the questionnaire
                10,    # MIN_PROP = 10  --> Useful for steering the probabilities
                0.1,   # pos_link = 0.1  --> Possibility to form a link on the innitial matrix
                5000,  # max_iterations = 5000
                2]     # ALPHA = 2

settings_lst = [0.05,
                0.15,    # RHO
                0.65,
                0.175,
                0.035,
                0.1,
                0.1,
                0.2]
DELTA, RHO, GAMMA, C, SIGMA, B1, B2, B3 = settings_lst
MAX_FRIEND, MIN_PROP, pos_link, max_iterations, ALPHA = constant_lst


class ConnectionMatrix:
    def __init__(self, n, p_link_0):
        self.minimal = 1
        self.n = n
        self.link_prop = p_link_0

        self.g = self.make_g_0()
        self.age = np.zeros((n, n)) + 1
        self.age_update()

    def make_g_0(self):
        """
        Initialize the first connections using the probabilitie of linking: link_prop
        """
        g_0 = np.random.choice([0, 1], size=(self.n, self.n),
                               p=[1 - self.link_prop,
                               self.link_prop])

        # Since you can not link to yourself, the diagonal needs to be zero
        np.fill_diagonal(g_0, 0)
        return g_0

    def age_update(self):
        '''
        Not used yet
        '''
        self.age *= self.g
        self.age += self.g


class NetworkModel:

    def __init__(self, g, n, X, pos_X, spatial=False):
        self.n = n
        self.pos_X = pos_X
        self.g = g
      #  self.triads = np.diagonal(np.linalg.matrix_power(self.g, 1))
        self.spatial = spatial
        self.X = X
        self.U = self.make_pre_cal_U()
        self.P = self.make_prop()

        self.indexes = list(range(n))
        self.g_sequence = None

    def make_prop(self):
        '''
        This makes
        '''
        # Make room
        prop = np.zeros((self.n, self.n))

        # Loop over the person and their peers
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    prop[i, j] = 0
                else:
                    prop[i, j] = self.U[i, j] + MIN_PROP

            # Normalize
            prop[i, :] = prop[i, :] / np.sum(prop[i, :])

        return prop

    def make_pre_cal_U(self):
        DELTA, RHO, GAMMA, C, SIGMA, B1, B2, B3 = settings_lst
        """ Make the U matrix for the entire system
        """
        # Setup U
        pre_cal_u = np.zeros((self.n, self.n))

        race = list(self.X['race'])
        sex = list(self.X['sex'])
        grade = list(self.X['grade'])

        try:
            new = list(self.X['new'])
            # Fill U
            for i in range(self.n) :
                for j in range(self.n) :
                    pre_cal_u[i, j] = math.exp(- B1 * abs(sex[i] - sex[j])
                                               - B2 * abs(grade[i] - grade[j])
                                               - B3 * (0 if race[i] == race[j] else 1)
                                               - ((B1+B2+B3)/3) * abs(new[i] - new[j]))
        except:
            # Fill U
            for i in range(self.n):
                for j in range(self.n):
                    pre_cal_u[i, j] = math.exp(- B1 * abs(sex[i] - sex[j])
                                               - B2 * abs(grade[i] - grade[j])
                                               - B3 * (0 if race[i] == race[j] else 1))


        return pre_cal_u

    def U_of_matrix(self, i):
        """ Returns the full utility of agent i given the current network structure
        g and the matrix of characteristics X """
        # degree, connection gain and cost calculations
        d_i = self.g.g[i].sum()
        direct_u = np.sum(self.g.g[i] * self.U[i])
        mutual_u = np.sum(self.g.g[i] * self.g.g.T[i] * self.U[i])

        # indirect connection gain
        a = (self.g.g.T.dot(self.g.g[i, :]) * self.U)[i]
        a[i] = 0
        indirect_u = np.sum(a)
        #triadic connection gain
        try:
            triads_u = self.triads[i] * self.U[i]
        except:
            triads_u = self.U[i]
        #total utility
        return direct_u + GAMMA * mutual_u + DELTA * indirect_u - d_i ** ALPHA * C # + RHO * triads_u

    def V_total(self, i):
        #Set the potential of the components initially to 0
        d_i_V = 0
        direct_V = 0
        mutual_V = 0
        indirect_V = 0
        triads_V = 0
        #Let's sum over the different components to get the potential of those components
        for i in range(len(self.g.g)):
            d_i = self.g.g[i].sum()
            direct_u = np.sum(self.g.g[i] * self.U[i])
            mutual_u = np.sum(self.g.g[i] * self.g.g.T[i] * self.U[i])
            #indirect
            a = (self.g.g.T.dot(self.g.g[i, :]) * self.U)[i]
            a[i] = 0
            indirect_u = np.sum(a)

            triads_u = self.triads[i] * self.U[i]
            #here the summing up really begins
            direct_V += direct_u
            mutual_V += mutual_u
            triads_V += triads_u
            indirect_V += indirect_u
            d_i_V += d_i ** ALPHA * C
        #total potential. It will be called in def run
        V_tot = direct_V + GAMMA * mutual_V + DELTA * indirect_V + RHO * triads_V - d_i_V

    def step(self, i=-1, r1=-1):
        """ Revises the tie of agents i and r1 and updates g accordingly  """

        # Add noise and shuffle indexes
        eps = np.random.normal(scale=SIGMA, size=self.n*2)
        np.random.shuffle(self.indexes)

        if i == r1 == -1 and not self.spatial:
            for i in self.indexes:
                # Choose new connection
                r1 = i
                while r1 == i:
                    r1 = np.random.choice(self.indexes, p=self.P[i])

                # find value for new connection and removed connection
                self.g.g[i, r1] = 0
                U_without = self.U_of_matrix(i) + eps[i]

                self.g.g[i, r1] = 1
                U_with = self.U_of_matrix(i) + eps[-i]

                # Evaluate better option
                if U_without > U_with:
                    self.g.g[i, r1] = 0
                else:
                    self.g.g[i, r1] = 1
        elif i != r1 and self.spatial:
            # find value for new connection and removed connection
            self.g.g[i, r1] = 0
            U_without = self.U_of_matrix(i) + eps[i]

            self.g.g[i, r1] = 1
            U_with = self.U_of_matrix(i) + eps[-i]

            # Evaluate better option
            if U_without > U_with:
                self.g.g[i, r1] = 0
            else:
                self.g.g[i, r1] = 1

        # self.triads= np.diagonal(np.linalg.matrix_power(self.g, 3))

    def save2pickle(self, pickle_name):
        """
        Saves data from the simulation to a pickle file.
        :param pickle_name: Name of pickle file
        """
        individuals_friendships_utilities = [self.X, self.g.g, self.U]
        pickle.dump(individuals_friendships_utilities, open(pickle_name, "wb"))

    def run(self, total_time, t_plot=0):
        """
        Run the ABM simulation
        :param total_time: Maximum iterations
        :param t_plot: Produce plots every t_plot periods
        :return: None (use plot to show results or save to save them)
        """
        if t_plot == 0:
            t_plot = total_time

        self.g_sequence = [self.g.g]

        for t in range(1, total_time):
            # Perform a step and append the new network
            self.step()
            self.g_sequence.append(self.g.g.copy())

            # Get rid of all g_(t-k) with k >= n_zeros_conv
            # this keeps only the g matrices in memory that we need to calculate
            # the convergence criterion (avoids memory issues).
            if t > n_zeros_conv:
                self.g_sequence = self.g_sequence[1:]

                # Convergence rule: Count the number of changes in g in the
                # n_zeros_conv last periods
                last_changes = [np.linalg.norm((self.g_sequence[-k-2] -
                        self.g_sequence[-1]), ord=1) for k in range(n_zeros_conv)]

                # stop if there were 0 changes in the last n_zeros_conv periods
                if t > n_zeros_conv and all(last_changes==np.zeros(n_zeros_conv)):
                    print('Converged after {} steps'.format(t+1))
                    #print the potential at convergence as well
                    print('The total potential of the network at convergence was:', V_tot)
                    break

            # Print a statement if simulation did not converge
            if t == total_time-1:
                print('No convergence reached in {} steps'.format(total_time))

            # Produce a plot and diagnostics every t_plot steps
            if t % t_plot == 0:
                print("degree:", np.sum(self.g.g))
                self.plot_network()

        return t

    def rank(self):
        """
        Rank the connections you have in order to reduce the connection matrix
        to max_friend male and female friends for each agent
        """
        print('ranking...')

        value_of_con = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                old = self.g.g[i][j]

                self.g.g[i][j] = 0
                not_conn = self.U_of_matrix(i)

                self.g.g[i][j] = 1
                conn = self.U_of_matrix(i)

                self.g.g[i][j] = old
                value_of_con[i][j] = not_conn - conn

        sex = list(self.X['sex'])

        def sort_value(x):
            return x[1]

        for i in range(self.n):
            male = []
            female = []
            for j in range(self.n):
                if self.g.g[i][j] != 0:
                    try:
                        if sex[j] == 1:
                            male.append((j, value_of_con[i][j]))
                        else:
                            female.append((j, value_of_con[i][j]))
                    except:
                        pass
            try:
                female.sort(key=sort_value, reverse=True)
                male.sort(key=sort_value, reverse=True)

                remain = female[:MAX_FRIEND]+male[:MAX_FRIEND]
                ind, score = zip(*remain)
            except:
                continue

            for friend in range(self.n):
                try:
                    if friend not in ind:
                        self.g.g[i][friend] = 0
                except:
                    pass


def analyse_network(connectivity, characteristics) :
    """
        Calculate mean and standard deviation (output in tuples [mean,sd]) of following measures:

        - degree
        - mut_prop
        - cluster_coef
        - segreg_ind    > list with length of the amount of characteristics (3 for sex, race, grade)

        INPUT:
        - connectivity matrix with row-students nominating column-students as friends
        - characteristics matrix with row per student, with integers indicating every group for each characteristic (sex, race, grade)
    """

    charr = np.zeros((characteristics.shape[0], 3))

    for i in range(3) :
        charr[:, i] = characteristics.iloc[:, i]

    characteristics = charr

    # get amount of nodes and list of out going dyads for every individual
    nodes = connectivity.shape[0]
    out_d = np.count_nonzero(connectivity, axis=1)

    # determine degree nodes (outgoing connections)
    mean_degree = np.mean(out_d)
    std_degree = np.std(out_d)
    degree = [mean_degree, std_degree]

    # determine the mutual dyads proportion
    # create matrix with 2's on mutual dyads, 1's on asymmetric dyads and count occurrence
    added_up = connectivity + np.transpose(connectivity)
    mutual_d = np.count_nonzero(added_up == 2, axis=1)
    mut_prop = (mutual_d / out_d) if len(list(out_d)) > 0 else 0
    # remove 'nan' individuals (with no out-going connections) from list
    mut_prop = [value for value in mut_prop if not math.isnan(value)]
    # calculate mean+std mutual dyads proportion
    mean_mut_prop = np.mean(mut_prop)
    std_mut_prop = np.std(mut_prop)
    mut_prop = [mean_mut_prop, std_mut_prop]

    # determine the local clustering coefficient
    clustering_coefficients = []
    for n_node, connections in enumerate(connectivity) :
        # the amount of neighbours each node has
        n_neighbours = np.sum(connectivity[n_node])
        # only consider nodes with at least 2 neighbours
        if n_neighbours >= 2 :
            # matrix of the nodes that are both neighbours of the node considered
            neighbour_matrix = np.dot(np.transpose([connectivity[n_node]]), [connectivity[n_node]])
            # the amount of connections between neighbours
            neighbour_connections = np.sum(connectivity * neighbour_matrix)
            # the amount of connections between neighbours divided by the possible amount of connections
            clustering_coefficients.append(neighbour_connections / (n_neighbours * (n_neighbours - 1)))
    # calculate mean+std clustering coefficient
    mean_cluster_coef = np.mean(clustering_coefficients)
    std_cluster_coef = np.std(clustering_coefficients)
    cluster_coef = [mean_cluster_coef, std_cluster_coef]

    # determine the segregation index per characteristic (sex, race, grade)
    segreg_ind = []
    # iterate through different characteristics (sex, race, grade)
    for i in range(characteristics.shape[1]) :
        # get different groups of this characteristic in dataset
        characs = sorted(list(set(characteristics[:, i])))
        amount = len(characs)
        # for every characteristic own tuple for mean and std
        segreg_ind_charac = []
        # iterate through different groups of this characteristic
        for j in range(amount) :
            # indicate indices of members this group and save size group
            indices = np.where(characteristics[:, i] == characs[j])[0]
            # calculate ratio out-group individuals
            ratio_diff = 1 - len(indices) / nodes
            # create a submatrix of all nominations from this group and save amount
            submat_trait = connectivity[np.ix_(indices, )]
            # create submatrix outgoing connections to individuals different group
            mask = np.ones(connectivity.shape[0], np.bool)
            mask[indices] = 0
            submat_diff = submat_trait[:, mask]
            # calculate segregation index per individual of this group for this characteristic
            for ind in range(len(indices)) :
                expect_out = submat_trait[ind].sum() * ratio_diff
                observ_out = submat_diff[ind].sum()
                seg_ind = (expect_out - observ_out) / expect_out if expect_out else 0
                if seg_ind < -1 :
                    seg_ind = -1
                segreg_ind_charac.append(seg_ind)
        # remove 'nan' individuals from list
        segreg_ind_charac = [value for value in segreg_ind_charac if not math.isnan(value)]
        # calculate mean+std segregation index this characteristic
        mean_segreg_ind_charac = np.mean(segreg_ind_charac)
        std_segreg_ind_charac = np.std(segreg_ind_charac)
        segreg_ind.append([mean_segreg_ind_charac, std_segreg_ind_charac])

        # TRIADS

    #calculate a list with triads (the diaginonal of g^3)
    triads = np.diagonal(np.linalg.matrix_power(connectivity, 3))

    total_degree = 0
    total_triads = 0

    for i in range(len(connectivity)):
        #degree per node
        d_i = connectivity[i].sum()
        #number of triads per node
        number_of_triads = (triads[i])
        total_degree += d_i
        total_triads += number_of_triads
        #print(degree)
        #print(number_of_triads)

    triad_prop = total_triads/total_degree
    triad_lst = [triads, triad_prop, total_triads, total_degree]

    return degree, mut_prop, cluster_coef, segreg_ind, triad_lst

def run(settings, X):
    '''
    :param settings: list of settings (=[DELTA, GAMMA, C, SIGMA, B1, B2, B3])
    :param schools:  lsit of schools based on index number
    :return: The outcome of the analyse function in a dict, ordered the same way as schools.
    '''
    global DELTA, GAMMA, RHO, C, B1, B2, B3, SIGMA, ALPHA, MIN_PROP, pos_link, MINIMAL, MAX_FRIEND, n_zeros_conv

    DELTA, RHO, GAMMA, C, SIGMA, B1, B2, B3 = settings

    # Constants
    MAX_FRIEND = 5  # maximum of 5 male and 5 female friends in the questionaire
    MIN_PROP = 1000  # 0 means meeting similar individuals more likely,
    # high values -> approaching uniform probabilities

    # number of agents
    n_agents = len(X)

    # Possibility to form a link on the initial matrix
    avg_initial_links = 5 # desired average degree in initial network
    pos_link = avg_initial_links/n_agents

    max_iterations = 5000
    ALPHA = 2 # makes cost quadratic in degree

    # Convergenge
    n_zeros_conv = 3  # periods of 0 change before convergence is declared



    # Make model and connection matrix
    g = ConnectionMatrix(n_agents, pos_link)
    M = NetworkModel(g, n_agents, X, [])

    # Run and rank the model
    M.run(max_iterations)
    M.rank()  # chooses best 5 female and male friends

    return M.g.g
