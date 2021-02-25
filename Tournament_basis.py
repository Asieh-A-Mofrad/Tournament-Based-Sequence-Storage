#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 10:03:31 2020

Last update: Feb. 25, 2021

@authors: Asieh & Samaneh

"""

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import pickle


class  Tournumant(object): # Ok!

    """
    Different types of structure will be assigned.

    """

    def __init__(self, parameter):

        if parameter['memory_type'] == 'Tournament_Winner':
            self.tournament = Tournament_Winner(parameter)
        elif parameter["memory_type"] == 'Tournament_Cache_Winner':
            self.tournament = Tournament_Cache_Winner(parameter)
        elif parameter["memory_type"] == 'Tournament_Explore_Winner':
            self.tournament = Tournament_Explore_Winner(parameter)
        elif parameter["memory_type"] == 'Tournament_Feedback_Winner':
            self.tournament = Tournament_Feedback_Winner(parameter)
        elif parameter["memory_type"] == 'Tournament_Backward_Winner':
            self.tournament = Tournament_Backward_Winner(parameter)


class Tournament_Winner(object): # Ok!

    """
    Tournament-based auto-associative neural network with the simpleast form of
    retrieval, i.e. choosing randomly from candidate set whenhever it in not unique
    """

    def __init__(self, parameter):

        self.c = parameter['c']
        self.k = parameter['k']
        self.L = parameter['L']
        self.r = parameter['r']
        self.Num_Seq = parameter['Num_Seq']
        self.Iter = parameter['Iter']
        self.l = 2**self.k
        self.name = parameter['memory_type']
        self.decision_record = {}
        self.decision_record_avg = {}
        self.active_list = {}


    def Learning_Set_Generator(self): # Ok!

        """
        This method produces a dictionary with the number of sequences, S, as keys
        which obtains from self.Num_Seq. Learning_set[S][i] is the ith
        random sequence of size self.L (L is the length of sequence)
        Each component is between 0 and self.l-1.
        All sequences are different in at least one component in
        the first r+1 components.
        """

        Learning_set = {}
        for S in self.Num_Seq:
            print(f'Learning set generated for; Size = {S}')
            Learning_set[S] = {}
            i = 0
            while i < S:
                L_c = np.random.randint(0, self.l-1, self.L).tolist()
                if L_c[:(self.r+1)] not in [Learning_set[S][t][:(self.r+1)] for
                                                  t in Learning_set[S].keys()]:
                    Learning_set[S][i] = L_c
                    i += 1

        return Learning_set


    def Testing_set_Generator(self): # Ok!

        """
        This method generates a dictationary with the number of sequences, S,
        as keys and a list of self.Iter indices as values for the testing phase.
        """

        Test_set_indx = {}
        for S in self.Num_Seq:
            if S < self.Iter:
                Test_set_indx[S] = np.random.randint(0, S-1, self.Iter).tolist()
            else:
                Test_set_indx[S] = random.sample(range(S), self.Iter)

        return Test_set_indx


    def Project_Message_to_Neuron(self, msg): # Ok!

        """
        This method gets a message (msg) then retruns the projection of each
        sub-message to the neurones in each cluster. Neurons are labelled
        from 0 to self.c*self.l
        """

        F = []
        t = 0
        for i in range(len(msg)):
            F.append(msg[i] + t*self.l)
            t = (t+1)%self.c

        return F


    def Learning(self, learning_set): # Ok!

        """
        This method gets a dictionary of patterns supposed to be stored.
        Creates a directed graph, establishes the edges for all input patterns
        and returns the final graph.
        """

        G = nx.DiGraph()
        G.add_nodes_from(range(self.c*self.l))
        for i in learning_set:
            Fanal = self.Project_Message_to_Neuron(learning_set[i])
            edges = []
            for j in range(self.L):
                R = min(self.r, self.L-j-1)
                for t in range(1, R+1):
                    edges.append((Fanal[j], Fanal[j+t]))
            G.add_edges_from(edges)

        return G


    def Candidate_Set_Generator(self, G, i, Active): # Ok!

        '''
        This methos gets a set of activated neurons in self.r previus clusters
        and the trained network G.
        Returns a set of neurons in cluster i%c that have maximum degree and
         most likely belong to the pattern.
        '''

        active = Active.copy()
        Candid = set(range((i%self.c)*self.l, (i%self.c + 1)*self.l))
        active |= Candid
        H = nx.DiGraph()
        H = G.subgraph(active)
        v_max = np.max([H.in_degree(j) for j in Candid])
        if v_max < len(Active):
            self.decision_record['candidate_set_error'] += 1

        if v_max == 0: # F
            return Candid
        Candidate = set()
        for j in Candid:
            if H.in_degree(j) == v_max:
                Candidate |= set([j])

        return Candidate


    def Retrieval_Tournament(self, Seq, G, debug = False): # Ok!

        '''
        This method uses the first r component of a previously learnt sequence,
        Seq, together with learnt network G and returns a complete sequence
        '''

        self.active_list = {}
        cue = self.Project_Message_to_Neuron(Seq)[0 : self.r]
        for i in range(self.r):
            self.active_list[i] = cue[i]

        for i in range(self.r, self.L):
            Active = set([self.active_list[j] for j in range(i-self.r, i)])
            Candidate = set()
            Candidate = self.Candidate_Set_Generator(G, i, Active)

            if len(Candidate) >= 1:
                ii = random.sample(Candidate, 1)[0]
                self.active_list[i] = ii
                self.decision_record['total'] += 1
                if len(Candidate) > 1:
                    self.decision_record['random'] += 1
                    if debug == True:
                        print('NO UNIQUE CHOICE FOR NEURON ACTIVATIONS. ONE IS CHOSEN RANDOMELY.')

        return self.active_list


    def decision_record_ini(self):

        """
        To remove previous rounds data for new learning set
        """

        if self.decision_record_avg == {}:
            self.decision_record_avg['random'] = {}
            self.decision_record_avg['total'] = {}
            self.decision_record_avg['candidate_set_error'] = {}

        self.decision_record['random'] = 0
        self.decision_record['total'] = 0
        self.decision_record['candidate_set_error'] = 0


    def Test_Retrieval(self, result_no): # ok!

        '''
        This is the main method that:
            first: read the Lernin_set and Testing_index_set, saved in results folder
            then: generate a network and train it for each learning set of size S
            afterwards: retrieve the sequences accoring to the testing_index_set
            and compute the Sequence Error Rate (SER) and  Component Error Rate (CER)
            finally: Produce average SER and CER and save them in the result folder
        '''

        SER = {}
        CER = {}
        Learning_set = pickle.load( open( f'results/Learning_set_basis_{result_no}.p', "rb" ))
        Testing_index_set = pickle.load( open( f'results/Testing_set_basis_{result_no}.p', "rb" ))

        for S in self.Num_Seq:
            print(f'------------\n Retrieval in process for Learning set Size = {S} \n')
            ser = 0
            cer = 0
            self.decision_record_ini()

            Learn_set = Learning_set[S]
            G = self.Learning(Learn_set)

            for t in range(self.Iter):
                S_inx = Testing_index_set[S][t]
                Seq = Learn_set[S_inx]
                ss = self.Project_Message_to_Neuron(Seq)
                Retr = self.Retrieval_Tournament(Seq, G)
                Retr_s = pd.Series(Retr).sort_index()
                Error = np.array(Retr_s) - np.array(ss)
                err = np.count_nonzero(Error)
                if err != 0:
                    ser += 1
                    cer += err

            SER[S] = ser/(self.Iter)
            CER[S] = cer/(self.Iter*(self.L- self.r))
            for d_type in self.decision_record.keys():
                if type(self.decision_record[d_type]) != type(dict()):
                    self.decision_record_avg[d_type][S] = \
                    self.decision_record[d_type] / self.Iter
                else:
                    for exp_no in self.decision_record[d_type].keys():
                        self.decision_record_avg[d_type][exp_no][S] = \
                        self.decision_record[d_type][exp_no] / self.Iter

        result = [SER, CER, self.decision_record_avg]
        file_name = f'results/{self.name}_{result_no}_{self.Iter}.p'
        pickle.dump(result, open( file_name, "wb" ))
        return SER, CER


# Class with ID: 2
class Tournament_Cache_Winner(Tournament_Winner): #

    """
    This is to use cache data for random choices in retrieval whenhever a unique
    candidate is not nominated in a cluster.
    """

    def __init__(self, parameter): #

             super(Tournament_Cache_Winner, self).__init__(parameter)
             self.random_cache = {}

    def Candidate_Set_Generator(self, G, i, Active): # ok!

        '''
        This methos gets a set of activated neurons in self.r previus clusters
        and the trained network G.
        Returns a set of neurons in cluster i%c that have maximum degree and
         most likely belong to the pattern.
        '''

        active = Active.copy()
        Candid = set(range((i%self.c)*self.l, (i%self.c + 1)*self.l))
        active |= Candid
        H = nx.DiGraph()
        H = G.subgraph(active)
        v_max = np.max([H.in_degree(j) for j in Candid])
        if v_max < len(Active):
            for j in range(i - self.r, i):
                if len(self.random_cache[j]) > 0:
                    return j
            self.decision_record['candidate_set_error'] += 1

        if v_max == 0: # F
            return Candid
        Candidate = set()
        for j in Candid:
            if H.in_degree(j) == v_max:
                Candidate |= set([j])

        return Candidate


    def Retrieval_Tournament(self, Seq, G, debug = False): # Ok!

        '''
        This method uses the first r component of a previously learnt sequence,
        Seq, together with learnt network G and returns a complete sequence
        '''

        self.active_list = {}
        cue = self.Project_Message_to_Neuron(Seq)[0 : self.r]
        for i in range(self.r):
            self.active_list[i] = cue[i]
            self.random_cache[i] = set()

        i = self.r
        while i < self.L:
            self.random_cache[i] = set()
            Candidate = set()
            Active = set([self.active_list[j] for j in range(i-self.r, i)])
            Candidate = self.Candidate_Set_Generator(G, i, Active)

            self.decision_record['total'] += 1
            if type(Candidate) == int:
                self.decision_record['random_revised'] += 1
                i = Candidate
                ii = random.sample(self.random_cache[i], 1)[0]
                self.random_cache[i] -= set([ii])
                self.active_list[i] = ii
            elif len(Candidate) >= 1:
                ii = random.sample(Candidate, 1)[0]
                self.active_list[i] = ii
                if len(Candidate) > 1:
                    self.decision_record['random'] += 1
                    self.random_cache[i] = Candidate - set([ii])
                    if debug == True:
                        print('NO UNIQUE CHOICE FOR NEURON ACTIVATIONS. ONE IS CHOSEN RANDOMELY.')
            self.random_cache[i-self.r] = set()
            i += 1

        return self.active_list

    def decision_record_ini(self):

        """
        To remove previous rounds data for new learning set
        """

        if self.decision_record_avg == {}:
            self.decision_record_avg['random'] = {}
            self.decision_record_avg['random_revised'] = {}
            self.decision_record_avg['total'] = {}
            self.decision_record_avg['candidate_set_error'] = {}

        self.decision_record['random_revised'] = 0
        self.decision_record['random'] = 0
        self.decision_record['total'] = 0
        self.decision_record['candidate_set_error'] = 0


# Class, with ID: 3
class Tournament_Explore_Winner(Tournament_Winner):

    """
    This is to use exploration in the retrieval whenhever a unique
    candidate is not nominated in a cluster.
    """

    def __init__(self, parameter):

             super(Tournament_Explore_Winner, self).__init__(parameter)
             self.r_explor = parameter['r_explore']


    def Retrieval_Tournament(self, Seq, G, debug = False): # Ok!

        '''
        This method uses the first r component of a previously learnt sequence,
        Seq, together with learnt network G and returns a complete sequence.
        It uses exploration technique to reduce retrieval ambigutiy.
        '''

        self.active_list = {} # The keys are 0 to L-1
        cue = self.Project_Message_to_Neuron(Seq)[0 : self.r]
        for i in range(self.r):
            self.active_list[i] = cue[i]

        for i in range(self.r, self.L):
            Active = set([self.active_list[j] for j in range(i-self.r, i)])
            Candidate = set()
            Candidate = self.Candidate_Set_Generator(G, i, Active)

            self.decision_record['total'] += 1
            if len(Candidate) == 1:
                for ii in Candidate:
                    self.active_list[i] = ii
                    Candidate = set()

            if len(Candidate) > 1:
                self.decision_record['random_explore'] += 1
                if debug == True:
                    print('NO UNIQUE CHOICE FOR NEURON ACTIVATIONS. EXPLORATION IS START.')
                R_explor = min(self.r_explor, self.L - i - 1)
                if R_explor > 0:
                    Candidate = self.Exploration(G, Candidate, i, R_explor)

                if len(Candidate) > 1:
                    self.decision_record['random'] += 1

                ii = random.sample(Candidate, 1)[0]
                self.active_list[i] = ii
                Candidate = set()

        return self.active_list


    def Exploration(self, G, candidate, i, R_explor, debug = False):  # Ok!

        '''
        G is the graph from learning phase. candidate_set contains more than one neuron with the same degree r.
        active_list is a active neurons in the previous steps. r_explor must be less than r, which is the
        number of clusters after i, which are used for exploration and limits the candidate set size.
        '''

        if debug == True:
            print('Exploration Phase starts')
        candidate_exp = set()
        candidate_set = candidate.copy()
        candidate_exp |= candidate_set # contains neurons in a cluster that might be activated and make a clique
        clq_explor = {}

        for j in range(1, R_explor + 1):
            candid = set()
            ii = i + j # ii must be less than L
            rr = self.r - j ## rr is the number of active neurons that has connections to ii
            Active = set([self.active_list[t] for t in range(i-rr, i)]) ### range(i-rr+1, i)
            candidate_exp |= self.Candidate_Set_Generator(G, ii, Active) # set of candidate
                                                                         #neurons in cluster i+j
            H_exp = nx.DiGraph()
            H_exp = G.subgraph(candidate_exp)

            max_degree = np.max([H_exp.out_degree(jj) for jj in candidate_set])
            if max_degree < j:
                self.decision_record['explore_max_failed'] += 1
                if debug == True:
                    print(f'EXPLORATION FAILED AT {j}TH STEP since {max_degree} < {j}')

            degree = min(max_degree, j)
            New_candid = set([cnd for cnd in candidate_set if H_exp.out_degree(cnd) >= degree])
            if len(New_candid) == 0:
                if debug == True:
                    print(f'EXPLORATION FAILED AT {j}TH STEP')
                self.decision_record['explore_forward_failed'] += 1
                return candidate_set
            if len(New_candid) == 1:
                if debug == True:
                    print(f'Unique candidate after {j} steps with Forward Technique')
                self.decision_record['explore_forward'][j] += 1
                return New_candid

            H_exp_undirect = nx.to_undirected(H_exp)
            Tournament_exp = nx.cliques_containing_node(H_exp_undirect, nodes = list(New_candid))
            clq_explor[j] = set()
            for k, v in Tournament_exp.items():
                A_q = [set(Tournament_exp[k][t]) for t in range(len(v)) if len(Tournament_exp[k][t]) == j+1]
                clq_explor[j] |= set(tuple(t) for t in A_q) # All possible cliques in cluster
            for node in New_candid:
                for cc in clq_explor[j]: # cc is a clique
                    if node in cc:
                        candid |= {node}

            if len(candid) == 0:
                if debug == True:
                    print('Exploration Failed!')
                self.decision_record['explore_clique_failed'] += 1
                return New_candid
            else:
                candidate_set = candid
                if len(candidate_set) == 1:
                    if debug == True:
                        print(f'Unique candidate after {j} steps with Clique Technique')

                    self.decision_record['explore_clique'][j] += 1
                    return candidate_set

        if len(candidate_set) < len(New_candid):
            self.decision_record['explore_reduce'] += 1
            if debug == True:
                print('Reduced the size of Candidate set')
        return candidate_set


    def decision_record_ini(self):

        """
        To remove previous rounds data for new learning set
        """

        if self.decision_record_avg == {}:
            self.decision_record_avg['random'] = {}
            self.decision_record_avg['total'] = {}
            self.decision_record_avg['candidate_set_error'] = {}
            ### For explorarion
            self.decision_record_avg['random_explore'] = {}
            self.decision_record['explore_forward'] = {}
            self.decision_record_avg['explore_forward'] = {}
            self.decision_record['explore_clique'] = {}
            self.decision_record_avg['explore_clique'] = {}
            self.decision_record_avg['explore_reduce'] = {}
            self.decision_record_avg['explore_forward_failed'] = {}
            self.decision_record_avg['explore_clique_failed'] = {}
            self.decision_record_avg['explore_max_failed'] = {}

        self.decision_record['random'] = 0
        self.decision_record['total'] = 0
        self.decision_record['candidate_set_error'] = 0
        ### For explorarion
        self.decision_record['random_explore'] = 0
        self.decision_record['explore_reduce'] = 0
        self.decision_record['explore_forward_failed'] = 0
        self.decision_record['explore_clique_failed'] = 0
        self.decision_record['explore_max_failed'] = 0


        for r_exp in range(1, self.r_explor + 1):
            self.decision_record['explore_forward'][r_exp] = 0
            self.decision_record['explore_clique'][r_exp] = 0

        if self.decision_record_avg['explore_forward'] == {}:
            for r_exp in range(1, self.r_explor + 1):
                self.decision_record_avg['explore_forward'][r_exp] = {}
                self.decision_record_avg['explore_clique'][r_exp] = {}


# Class with ID: 4
class Tournament_Feedback_Winner(Tournament_Winner): # Ok!

    """
    This is to use feedback in the learning and retrieval whenhever a unique
    candidate is not nominated in a cluster.
    """

    def __init__(self, parameter):
        super(Tournament_Feedback_Winner, self).__init__(parameter)
        self.r_fdbk = parameter['r_fdbk']
        self.r_fwd = self.r - self.r_fdbk

    def Learning(self, learning_set): # Ok!

        """
        This function gets a dictionary of patterns supposed to be stored,
        as the input and for each pattern establish the edges
        and returns the graph with all edges.
        """

        G = nx.DiGraph()
        G.add_nodes_from(range(self.c*self.l))
        for i in learning_set:
            Fanal = self.Project_Message_to_Neuron(learning_set[i])
            edges = []
            for j in range(self.L):
                R = min(self.r_fwd, self.L - j - 1)
                for t in range(1, R+1):
                    edges.append((Fanal[j], Fanal[j+t]))

                if j < self.L- self.r_fwd - 1:
                    R_feed = min(self.r_fdbk , self.L - j - 1 - self.r_fwd)
                    for tt in range(1, R_feed + 1):
                        edges.append((Fanal[j + self.r_fwd + tt], Fanal[j]))

            G.add_edges_from(edges)

        return G


    def Feedback(self, G, candidate, i, debug = False): # Ok!

        '''
        This function gets a candidate set in cluster i%c and returns a unique
        node or an updated set by using the backward connections.
        '''

        candidate_set = candidate.copy()
        candidate_fdbk = set()
        candidate_fdbk |= candidate_set # contains neurons in a cluster that might be activated and make a clique
        active_fdbk = set([self.active_list[j] for j in range(i - self.r, i - self.r_fwd)])
        H_fdbk = nx.DiGraph()
        candidate_fdbk |= active_fdbk
        H_fdbk = G.subgraph(candidate_fdbk)

        max_degree = np.max([H_fdbk.out_degree(j) for j in candidate_set])


        if max_degree == 0:
            self.decision_record['feedback_failed'] += 1
            if debug == True:
                print(' FAILED! All Candidates are Wrong Based on the Feedback')
            return candidate_set
        elif max_degree < self.r_fdbk:
            self.decision_record['backward_unique_failed'] += 1
            if debug == True:
                print('FAILED! max_degree is less than r_fdbk')

        degree = min(max_degree, self.r_fdbk )
        New_candid = set([cnd for cnd in candidate_set if H_fdbk.out_degree(cnd) >= degree])

        if len(New_candid) == 1  and max_degree == self.r_fdbk:
            self.decision_record['feedback_succeed'] += 1
            if debug == True:
                print('YESSS! FEEDBACK HELPS RETRIEVAL')
        elif len(New_candid) == len(candidate):
            if debug == True:
                print('NO! FEEDBACK DOES NOT HELPS AT ALL')
            self.decision_record['feedback_neutral'] += 1
        else:
            if debug == True:
                print('FEEDBACK REDUCES THE CANDIDATE SET SIZE')
            self.decision_record['feedback_reduced'] += 1

        return New_candid


    def Retrieval_Tournament(self, Seq, G, debug = False): # Ok!

        '''
        This function gets m_r, which is the first r component of a previousely learnt sequence.
        c is the number of clusters; L is the length of sequence, and k is the length of sub_sequences.
        l is the number of neurones per cluster and returns a complete sequence that is more likely to
        be the one starting with m_r.
        '''

        self.active_list = {} # The keys are 0 to L-1
        cue = self.Project_Message_to_Neuron(Seq)[0 : self.r]
        for i in range(self.r):
            self.active_list[i] = cue[i]

        for i in range(self.r, self.L):
            Active = set([self.active_list[j] for j in range(i - self.r_fwd, i)])
            Candidate = set()
            Candidate = self.Candidate_Set_Generator(G, i, Active)

            self.decision_record['total'] += 1
            if len(Candidate) == 1:
                for ii in Candidate:
                    self.active_list[i] = ii
                    Candidate = set()

            if len(Candidate) > 1:
                self.decision_record['random_feedback'] += 1
                if debug == True:
                    print('NO UNIQUE CHOICE FOR NEURON ACTIVATIONS. FEEDBACK LINKS MUST BE USED')

                Candidate = self.Feedback(G, Candidate, i)
                if len(Candidate) > 1:
                    self.decision_record['random'] += 1
                ii = random.sample(Candidate, 1)[0]
                self.active_list[i] = ii
                Candidate = set()

        return self.active_list


    def decision_record_ini(self):

        """
        To remove previous rounds data for new learning set
        """

        if self.decision_record_avg == {}:
            self.decision_record_avg['random'] = {}
            self.decision_record_avg['total'] = {}
            self.decision_record_avg['candidate_set_error'] = {}
            ### For feedback
            self.decision_record_avg['random_feedback'] = {}
            self.decision_record_avg['feedback_succeed'] = {}
            self.decision_record_avg['feedback_failed'] = {}
            self.decision_record_avg['feedback_neutral'] = {}
            self.decision_record_avg['feedback_reduced'] = {}
            self.decision_record_avg['backward_unique_failed'] = {}

        self.decision_record['random'] = 0
        self.decision_record['total'] = 0
        self.decision_record['candidate_set_error'] = 0
        ### For feedback
        self.decision_record['random_feedback'] = 0
        self.decision_record['feedback_succeed'] = 0
        self.decision_record['feedback_failed'] = 0
        self.decision_record['feedback_neutral'] = 0
        self.decision_record['feedback_reduced'] = 0
        self.decision_record['backward_unique_failed'] = 0


# Class with ID: 5
class Tournament_Backward_Winner(Tournament_Feedback_Winner): # Ok!

    """
    This is to use backward retrieval when we have feedback links
    """

    def __init__(self, parameter): #

             super(Tournament_Backward_Winner, self).__init__(parameter)


    def Forward(self, G, candidate, i, debug = False):

        '''
        This function gets a candidate set in cluster i%c and returns a unique node or the same set
        by using the backward connections.
        '''

        candidate_set = candidate.copy()
        candidate_fdbk = set()
        candidate_fdbk |= candidate_set # contains neurons in a cluster that might be activated and make a clique
        active_fdbk = set([self.active_list[j] for j in range(i + 1, i + self.r_fwd + 1)])
        H_fdbk = nx.DiGraph()
        candidate_fdbk |= active_fdbk
        H_fdbk = G.subgraph(candidate_fdbk)

        max_degree = np.max([H_fdbk.out_degree(j) for j in candidate_set])

        if max_degree == 0:
            self.decision_record['forward_failed'] += 1
            if debug == True:
                print('FAILED! All Candidates are Wrong Based on the Forward')
            return candidate_set

        elif max_degree < self.r_fwd:
            self.decision_record['forward_unique_failed'] += 1
            if debug == True:
                print('FAILED! max_degree is less than r_fwd')

        New_candid = set([cnd for cnd in candidate_set if H_fdbk.out_degree(cnd) == max_degree])

        if len(New_candid) == 1 and max_degree == self.r_fwd:
            self.decision_record['forward_succeed'] += 1
            if debug == True:
               print('YESSS! FORWARD HELPS RETRIEVAL')
        elif len(New_candid) == len(candidate):
            self.decision_record['forward_neutral'] += 1
            if debug == True:
                print('NO! Forward DOES NOT HELPS AT ALL')
        else:
            self.decision_record['forward_reduced'] += 1
            if debug == True:
                print('Forward REDUCES THE CANDIDATE SET SIZE')

        return New_candid


    def Retrieval_Tournament(self, Seq, G, debug = False ): # Ok!

        '''
        This function gets m_r, which is the last r component of a previousely learnt sequence.
        G is the learnt graph.
        '''

        self.active_list = {}
        t = self.L - self.r
        cue = self.Project_Message_to_Neuron(Seq)[t :]
        for i in range(t, self.L):
            self.active_list[i]= cue[i - t]

        for i in range(t -1, -1, -1):
            Candidate = set()
            active_fback = set()
            R_back = min(self.L, i + self.r + 1)
            active_fback = set([self.active_list[j] for j in range(i + self.r_fwd + 1, R_back)])
            Candidate = self.Candidate_Set_Generator(G, i, active_fback)

            self.decision_record['total'] += 1
            if len(Candidate) == 1:
                for ii in Candidate:
                    self.active_list[i] = ii
                    Candidate = set()

            if len(Candidate) > 1:
                self.decision_record['random_forward'] += 1
                if debug == True:
                    print('NO UNIQUE CHOICE FOR NEURON ACTIVATIONS. FORWARD LINKS MUST BE USED')
                Candidate = self.Forward(G, Candidate, i)
                if len(Candidate) > 1:
                    self.decision_record['random'] += 1
                ii = random.sample(Candidate, 1)[0]
                self.active_list[i] = ii
                Candidate = set()

        return self.active_list


    def decision_record_ini(self):

        """
        To remove previous rounds data for new learning set
        """

        if self.decision_record_avg == {}:
            self.decision_record_avg['random'] = {}
            self.decision_record_avg['total'] = {}
            self.decision_record_avg['candidate_set_error'] = {}
            ### For feedback
            self.decision_record_avg['random_forward'] = {}
            self.decision_record_avg['forward_succeed'] = {}
            self.decision_record_avg['forward_failed'] = {}
            self.decision_record_avg['forward_neutral'] = {}
            self.decision_record_avg['forward_reduced'] = {}
            self.decision_record_avg['forward_unique_failed'] = {}

        self.decision_record['random'] = 0
        self.decision_record['total'] = 0
        self.decision_record['candidate_set_error'] = 0
        ### For forward
        self.decision_record['random_forward'] = 0
        self.decision_record['forward_succeed'] = 0
        self.decision_record['forward_failed'] = 0
        self.decision_record['forward_neutral'] = 0
        self.decision_record['forward_reduced'] = 0
        self.decision_record['forward_unique_failed'] = 0

# Class for plotting
class Plot_Results(object):

    """
    This class plot the results which has been stored in 'file name' address
    """

    def __init__(self, file_name):

        self.file_name = file_name

    def Plot_Error(self):

        """
        reads the data and desired representations and plot them.
        """

        results = pickle.load( open(self.file_name, "rb" ))
        SER, CER, decision_record = results

        plt.title(self.file_name.split("/")[1][:-2], fontsize = 25)       
        plt.plot(list(SER.keys()), list(SER.values()),'r--', label = 'Sequence Error Rate')
        plt.plot(list(CER.keys()), list(CER.values()),'b--', label = 'Compnent Error Rate')
        plt.grid(True)
        plt.legend(fontsize = 20)
        plt.tick_params(labelsize = 16)
        plt.show()
        print('\n **SER results** \n')
        print(SER)
        print('\n **CER results** \n')
        print(CER)
        print ('\n------ **HERE is a record on the decision type when the candidate is not unique**-------\n' )
        for key , value in decision_record.items():
            print(key, value)
