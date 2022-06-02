from ABM_visualization import *
from ABM_classes import *
from fast_model import ConnectionMatrix, NetworkModel
import pickle

""" If not done yet, download X_list.pkl and g_list.pkl from the Google Drive link into the
    same directory as this file. """

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


def run_both(settings, constants, schools, nw_plot_steps=False, visualize=True):
    with open("X_list.pkl", mode="rb") as X_file, open("g_list.pkl", mode="rb") as g_file:
        big_x = pickle.load(X_file)
        g_matrix = pickle.load(g_file)

        global DELTA, GAMMA, C, B1, B2, B3, SIGMA, ALPHA, MIN_PROP, pos_link, MINIMAL, MAX_FRIEND, zeros, n_zeros_conv

        # Plot info
        #analyse_data = {'av_degree':[], 'clustering_coefficient':[], 'mut_prop':[],
         #               'segreg_ind0':[], 'segreg_ind1':[], 'segreg_ind2':[]}

        #_string = ['av_degree', 'mut_prop', 'clustering_coefficient', 'segreg_ind0', 'segreg_ind1', 'segreg_ind2']

        DELTA, RHO, GAMMA, C, SIGMA, B1, B2, B3 = settings
        MAX_FRIEND, MIN_PROP, pos_link, max_iterations, ALPHA = constants

        # Convergenge
        #n_zeros_conv = 3
        #zeros = np.zeros(n_zeros_conv)


        # Iterate over the given schools
        for school in schools:
            X = big_x[school]
            n_agents = len(X['sex'])

            # Plot variables
            if nw_plot_steps:
                possible_X = [i[0] for i in list(X.groupby(['sex', 'race']))]
                plot_step = int(25)
            else:
                possible_X = []
                plot_step = 0

            # Make model and connection matrix
            g = ConnectionMatrix(n_agents, pos_link)
            nw = NetworkModel(g, n_agents, X, possible_X, spatial=True)
            abm = SchoolModel(n_agents, nw)

            # Run and rank the model
            #M.run(max_iterations, plot_step)
            animate_model(abm, 500, 50)
            print("OI MATE")
            nw.rank()  # chooses best 5 female and male friends


run_both(settings_lst, constant_lst, [1], False, True)


