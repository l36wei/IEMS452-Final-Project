# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:10:38 2020
Combinatorial Optimization Project
Finding the maximum cardinality matching
@author: Linchuan Wei
@Date: 05/06/2020

"""
import gurobipy
import timeit
import os
import networkx as nx
from multiprocessing import Pool
import multiprocessing
import pathlib2
from pathlib2 import Path
from numpy import linalg as LA
import math 
import numpy as np 
from scipy.io import mmread
import random

data = mmread("C:\\Users\\Home\\graphs\\mark3jac020sc\\mark3jac020sc.mtx")
edges_list = list([tuple(row) for row in np.transpose(data.nonzero())])

[M, N] = data.shape

G = nx.Graph()

for i in range(M):
    G.add_node(i)
    
G.add_edges_from(edges_list)

G.remove_edges_from(nx.selfloop_edges(G))



M = []

def max_matching(G):
    if not G.edges():
        M = []
        return M
    else:
        degrees = [val for (node, val) in G.degree()]
        d = min(degrees)
        n_m = [node for (node, val) in G.degree() if val == d]
        if d == 0:
            for node in n_m:
                G.remove_node(node)
            M = max_matching(G)
            return M
        elif d == 1:
            v = random.choice(n_m)
            w = [n for n in G.neighbors(v) ][0]
            G.remove_node(v)
            G.remove_node(w)
            M = max_matching(G)
            M.append((v,w))
            return M
        elif d == 2:
            u = random.choice(n_m)
            neighbors = [n for n in G.neighbors(u) ]
            v = neighbors[0]
            w = neighbors[1]
            g = G.copy()
            g.remove_node(u)
            g= nx.identified_nodes(g, v, w)
            g.remove_edges_from(nx.selfloop_edges(g))
            M = max_matching(g)
            e = [(z,y) for (z, y) in M if z == v or y ==v ]
            if not e:
                M.append((u,v))
                return M
            else:
                if e[0][0] == v:
                    y = e[0][1]
                else:
                    y = e[0][0]
                if y in G.neighbors(v):
                    M.remove(e[0])
                    M.append((u,w))
                    M.append((y,v))
                    return M
                else:
                    M.remove(e[0])
                    M.append((v,u))
                    M.append((w,y))
                    return M
        else:
            while d > 2:
                 d_m = max(degrees)
                 n_m = [node for (node, val) in G.degree() if val == d_m]
                 v = random.choice(n_m)
                 neighbors = [n for n in G.neighbors(v)]
                 neighbor = random.choice(neighbors)
                 G.remove_edge(v,neighbor)
                 degrees = [val for (node, val) in G.degree()]
                 d = min(degrees)
            M = max_matching(G)
            return M
     