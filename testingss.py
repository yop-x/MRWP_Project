import pickle
import pandas as pd

schools = range(82) #[7, 16, 52, 69]


def class_distribution(school, cap):

    if n <= 0:
        return
    elif n > 2535:
        n = 2535

    nb_cr = n//cap + 1



with open("X_list.pkl", mode="rb") as X_file, open("g_list.pkl", mode="rb") as g_file:
    big_x = pickle.load(X_file)
    g_matrix = pickle.load(g_file)


    maximum = 0
    for school in schools:
        X = big_x[school]
        n_students = len(X["race"])
        print(X["race"])
        if n_students>maximum:
            maximum = n_students
        #print(maximum)
