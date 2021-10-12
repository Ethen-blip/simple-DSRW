#!/usr/bin/env python
# coding: utf-8

# simplest implementation of dsrw

import networkx as nx
import matplotlib.pyplot as plt
import random 
import numpy as np



graph_cmty_file = "./dataset/eucore/eucore_cmty.txt"     # high_node_x = 1.5
g = nx.read_edgelist("./dataset/eucore/eucore.txt")#, create_using=nx.DiGraph())


# hyper params for eucore, for different network situation, the hyper params should be diff
# some network concerened on nodes with high-connectivity, while other networks concerned on node with low-connectivity
# We need wo modify the hyper params to achieve a better performance
high_node_x = 1.5 
penaty_rate = 0.4

seeds = []
seed_cmty = set()

G_degree = 0
node_count = 0
for node in g.nodes():
    G_degree += g.degree(node)
    node_count += 1
G_avg_degree = G_degree/node_count

node_degree = {}
node_neighbor = {}
for node in g.nodes():
    node_degree[node] = g.degree(node)
    node_neighbor[node] = list(g.neighbors(node))


# In[2]:


G_avg_degree


# In[3]:


def normalize(d):
    nd = {}
    sum_d = 0
    for key in d.keys():
        sum_d += d[key]
    for key in d.keys():
        nd[key] = d[key]/sum_d
    return nd
    #return d


# In[4]:


def best_cut(rate_dict_result):
    list_x = list(rate_dict_result.items())
    sv = sorted(list_x, key=lambda x: x[1], reverse=True)
    S = set()
    length = min(len(seed_cmty), len(sv))
    print(" ======== show: ", sv[length-1])
    for i in range(length):
        S.add(sv[i][0])
    return S


# In[5]:


# use seed_cmty, and rate_dict get f1 score 
def F1_score(detect_set):
    if len(detect_set)==0 or len(seed_cmty)==0: return 0.
    correct_detect = detect_set & seed_cmty
    precision = len(correct_detect)/float(len(detect_set))
    recall = len(correct_detect)/float(len(seed_cmty))
    if precision != 0 and recall != 0:
        score = 2 * precision * recall / float(precision + recall)
    else:
        score = 0.
    return score


# In[6]:




def randomWalk_with_DA():
    # Parameters
    apl = 0.9
    det = 0.01
    tht = 0.005

    # Random Walk with Density Aware
    r_old = {}
    seeds_num = len(seeds)
    for node in seeds:
        r_old[node] = 1/seeds_num
        
    for i in range(30):
        r_new = {}
        for node_i in r_old.keys():
            if r_old[node_i] > tht and node_degree[node_i] != 0:
                total_degree = 0
                for node_j in g.neighbors(node_i):
                    total_degree += node_degree[node_j]
                    
                p_j_dict = {}
                for node_j in g.neighbors(node_i):
                    if g.degree(node_j) > high_node_x*G_avg_degree:
                        p_j_dict[node_j] = 1/node_degree[node_i] #+ node_degree[node_j]/total_degree
                    else:
                        p_j_dict[node_j] = (1+penaty_rate)/node_degree[node_i] # transition probability matrix fixed, sure converge
                p_j_dict_norm = normalize(p_j_dict)
                
                for node_j in g.neighbors(node_i):
                    p_j = p_j_dict_norm[node_j]
                    if g.degree(node_j) < G_avg_degree:
                        tmp_apl = apl # * g.degree(node_j) / G_avg_degree
                    else:
                        tmp_apl = apl
                    
                    if node_j in r_new.keys():
                        r_new[node_j] = tmp_apl*p_j*r_old[node_i] + r_new[node_j]
                    else:
                        r_new[node_j] = tmp_apl*p_j*r_old[node_i]
                    for seed in seeds:
                        if seed in r_new.keys():
                            r_new[seed] = (apl-tmp_apl)*p_j*r_old[node_i]/seeds_num + r_new[seed]
                        else:
                            r_new[seed] = (apl-tmp_apl)*p_j*r_old[node_i]/seeds_num
                        
        for seed in seeds:
            if seed in r_new.keys():
                r_new[seed] = (1-apl)/seeds_num + r_new[seed]
            else:
                r_new[seed] = (1-apl)/seeds_num
        r_new = normalize(r_new)
        r_old = r_new.copy()

    return r_old.copy()



def get_simulate(iter_num):
    communitys = []
    with open(graph_cmty_file, 'r') as f:
        for line in f.readlines():
            community = line.strip().split('\t')
            if len(community) > 15:# and len(community) < 50: 
                communitys.append(community)
    print("community num: ", len(communitys))
    
    f1_rwda = []
    avg_score2 = 0
    for i in range(iter_num):
        random_cmty_num = random.randint(0, len(communitys) - 1)
        community = communitys[random_cmty_num]   
        global seeds
        seeds = random.sample(community, 3)   # 3 seeds
        global seed_cmty 
        seed_cmty = set(community)
        print("seeds: ", seeds)
        print("community: ", seed_cmty)
        
        rwda_r = randomWalk_with_DA()
        
        s_rwda_rate = best_cut(rwda_r)
        
        score2 = F1_score(s_rwda_rate)
        avg_score2 = (score2-avg_score2)/(i+1) + avg_score2
        f1_rwda.append(score2)
        print("Simulate process: ", i+1, "/", iter_num, " RWDA: ", score2)
        print("Current PK score: ", i+1, "/", iter_num, " RWDA: ", avg_score2)
    
    avg_f1_rwda = np.mean(f1_rwda)
    print ("Iteration ", iter_num, " times: ")
    print ("==================================")
    print ("F1 score of RWDA", avg_f1_rwda)
    


# In[12]:


get_simulate(5000)

