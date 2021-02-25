#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 10:56:42 2020

Last update: Feb. 25, 2021

@authors: Asieh & Samaneh
"""

import numpy as np

def Initialization(result_no, structure_ID):
    Initial_values = Simulation_parameters(structure_ID)
    return Initial_values[result_no]


def Simulation_parameters(structure_ID):
    """
    Here the information for the simulation will be provided.
    """
    memory_type = {1 : 'Tournament_Winner',
                   2 : 'Tournament_Cache_Winner',
                   3 : 'Tournament_Explore_Winner',
                   4 : 'Tournament_Feedback_Winner',
                   5 : 'Tournament_Backward_Winner'
        }
    Seq_1 = np.linspace(10, 4000, 5, dtype = int)
    Seq_2 = np.linspace(4500, 10500, 40, dtype = int)
    Seq_3 = np.linspace(11000, 15000, 5, dtype = int)
    Initial_values = {
            1: {
                'memory_type' : memory_type[structure_ID],
                'c' : 20,
                'k' : 4,
                'L' : 40,
                'r' : 6, # total. The size of tournament is r + 1
                'r_fdbk' : 3, # c >= 2*r - r_fdbk +1
                'r_explor' : 2, # r_explore < r-1
                'Num_Seq': np.linspace(10, 50, 10, dtype = int),
                'Iter' : 20
                    },
            2: {
                'memory_type' : memory_type[structure_ID],
                'c' : 20,
                'k' : 8,
                'L' : 100,
                'r' : 12, # 8
                'r_fdbk' : 6,
                'r_explore' : 7,
                'Num_Seq': np.concatenate([Seq_1, Seq_2, Seq_3]),
                'Iter' : 100
                    },
            }
    return Initial_values
