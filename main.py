# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 12:57:41 2020

Last update: Feb. 25, 2021

@authors: Asieh & Samaneh
"""

import pickle
import time

import Initialization as ini
import Tournament_basis as Tour



result_no = 2 ## Change the parameters in Initialization.py file
structure_ID = 5  # Change the ID to choose a different structure. Options are 1, 2, 3, 4 and 5
file_name = None  # Give the file_name to just plot a previousely saved simulation
Learning_set_flag = True # change it to False if you don't want to generate a new learning set

if file_name == None:
    parameter = ini.Initialization(result_no, structure_ID)
    general_memory = Tour.Tournumant(parameter)
    Memory = general_memory.tournament
    if Learning_set_flag:
        Learning_set = Memory.Learning_Set_Generator()
        pickle.dump(Learning_set, open( f'results/Learning_set_basis_{result_no}.p', "wb" ))
        Testing_index = Memory.Testing_set_Generator()
        pickle.dump(Testing_index, open( f'results/Testing_set_basis_{result_no}.p', "wb" ))

    Start = time.time()
    Memory.Test_Retrieval(result_no)
    Duration = time.time()-Start
    print('\n---------------\n execution time', Duration)

    file_name = f'results/{parameter["memory_type"]}_{result_no}_{parameter["Iter"]}.p'
    print('results are saved in', file_name)

Tour.Plot_Results(file_name).Plot_Error()
