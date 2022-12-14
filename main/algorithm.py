# In this file there will be the main algorithm
import collections
import operator
import os
import time
import networkx as nx
import numpy as np
import scipy.io
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from karateclub import DeepWalk
from karateclub import node_embedding as ne
import math
import statistics
import argparse
import logging

logger = logging.getLogger(__name__)


def neighborhood(G, node, level):
    return nx.single_source_dijkstra_path_length(G, node, cutoff=level)


def load_adjacency_matrix(file, variable_name="network"):
    os.chdir("../datasets")
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    return data[variable_name]


def deepwalk_embedding(file, variable_name="network"):
    G = nx.from_scipy_sparse_matrix(load_adjacency_matrix(file, variable_name),
                                    create_using=nx.Graph)
    logger.info("DeepWalk embedding of %s network", file)
    start_time = time.time()
    model = DeepWalk(walk_length=200, dimensions=128, window_size=10)
    model.fit(G)
    embedding = model.get_embedding()
    print("------- %s seconds ---------" % (time.time() - start_time))
    return G, embedding


def precomputing_euclidean(embedding):
    start = time.time()
    distances = dict()
    for i in range(len(embedding)):
        for j in range(len(embedding)):
            alias = (i,j)
            distances[alias] = -np.linalg.norm(embedding[i]-embedding[j])
    logger.info("Precomputing euclidean distances between node embeddings in %s seconds" % (time.time()-start))
    return distances


def compute_k_core_values(G):
    start_time = time.time()
    G.remove_edges_from(nx.selfloop_edges(G))
    k_core = nx.core_number(G)
    logger.info("Computed K-Core values in %s seconds" % (time.time() - start_time))
    return k_core


# algorithm NLC(G,N) in identifying influential spreaders...
def nlc(args):
    G, deepwalk_graph = deepwalk_embedding(args.matfile, variable_name=args.matfile_variable_name)
    core_values = compute_k_core_values(G)
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    for i in G:
        neighbors = neighborhood(G, i, level=1)
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(- np.linalg.norm(deepwalk_graph[i] - deepwalk_graph[j])))
    # sorting in descending order the nodes based on their NLC index  values
        print(nlc_indexes[i])
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    print(list(nlc_indexes[:k]))


# def check(h, d):
#     f = 0
#     for i in h.nodes():
#         if (h.degree(i) <= d):
#             f = 1
#             break
#     return f
#
#
# def find_nodes(h, it):
#     s = []
#     for i in h.nodes():
#         if (h.degree(i) <= it):
#             s.append(i)
#     return s
#
#
# def kShell_values(h):
#     start = time.time()
#     it = 1
#     tmp = []
#     buckets = []
#
#     while (1):
#         flag = check(h, it)
#         if (flag == 0):
#             it += 1
#             buckets.append(tmp)
#             tmp = []
#         if (flag == 1):
#             node_set = find_nodes(h, it)
#             for each in node_set:
#                 h.remove_node(each)
#                 tmp.append(each)
#         if (h.number_of_nodes() == 0):
#             buckets.append(tmp)
#             break
#
#     core_values = dict()
#
#     value = 1
#
#     for b in buckets:
#         for n in b:
#             core_values[n] = value
#         value += 1
#     logger.info(("kShell in %s seconds") % (time.time()-start))
#     return core_values


def asp_s(H, G):
    n = H.number_of_nodes()
    asp_value = 0
    if not (nx.is_connected(H)):
        diameter = nx.diameter(G)
    for i in H:
        for j in H:
            if nx.has_path(H, source=i, target=j):
                asp_value += nx.shortest_path_length(H, source=i, target=j)
            else:
                asp_value += diameter
    asp_value = asp_value / (n * (n-1))
    return asp_value


def asp(H):
    n = H.number_of_nodes()
    asp_value = 0
    if not (nx.is_connected(H)):
        diameter = nx.diameter(H)
    for i in H:
        for j in H:
            if nx.has_path(H, source=i, target=j):
                asp_value += nx.shortest_path_length(H, source=i, target=j)
            else:
                asp_value += diameter
    asp_value = asp_value / (n * (n-1))
    return asp_value


def local_rasp(args):
    G = nx.from_scipy_sparse_matrix(load_adjacency_matrix(args.matfile, variable_name=args.matfile_variable_name),
                                    create_using=nx.Graph)
    lrasp = dict.fromkeys(list(G.nodes), 0)
    for i in G:
        neighbors = neighborhood(G, i, level=2)
        H = G.subgraph(neighbors.keys())
        if H is not None:
            H_i = H.copy()
            H_i.remove_node(i)
            if H_i is not None:
                asp_value_H = asp(H)
                lrasp[i] = abs(asp_s(H_i, H) - asp_value_H) / asp_value_H
    lrasp = dict(sorted(lrasp.items(), key=operator.itemgetter(1), reverse=True))
    print(lrasp)
    k = args.top
    print(list(lrasp.keys())[:k])


def nlc_modified(args):
    G = nx.from_scipy_sparse_matrix(load_adjacency_matrix(args.matfile, variable_name=args.matfile_variable_name),
                                    create_using=nx.Graph)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.matfile)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # core_values = kShell_values(G)
    # mean_core_values = statistics.mean(list(core_values.values()))
    # Computing degree centrality for nodes
    start_time2 = time.time()
    degree_centrality = nx.degree_centrality(G)
    mean_degreecentrality = statistics.mean(list(degree_centrality.values()))
    logger.info("Computing degree centrality in %s seconds" % (time.time() - start_time2))
    # Computing betweenness centrality for nodes
    #start_time3 = time.time()
    #beetweennes = nx.betweenness_centrality(G)
    #mean_beetweennes = statistics.mean(list(beetweennes.values()))
    #logger.info("Computing betweenness centrality in %s seconds" % (time.time() - start_time3))
    # Computing the NLC index of each node
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    logger.info("Computing NLC index of each node")
    for i in G:
        if degree_centrality[i] > mean_degreecentrality:
            neighbors = neighborhood(G, i, level=3)
            for j in neighbors:
                nlc_indexes[i] += (core_values[i] * math.exp(distances[i,j]))
    # sorting in descending order the nodes based on their NLC index  values
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    print(list(nlc_indexes.keys())[:k])


def nlc_modified_second(args):
    G = nx.from_scipy_sparse_matrix(load_adjacency_matrix(args.matfile, variable_name=args.matfile_variable_name),
                                    create_using=nx.Graph)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.matfile)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # core_values = kShell_values(G)
    # mean_core_values = statistics.mean(list(core_values.values()))
    # Computing degree centrality for nodes
    #start_time2 = time.time()
    #degree_centrality = nx.degree_centrality(G)
    #neighborhood_degree_centrality = dict.fromkeys(core_values.keys(), 0)
    #logger.info("Computing degree centrality in %s seconds" % (time.time() - start_time2))
    # Computing degree for each nodes
    start_time4 = time.time()
    degrees = {node: val for (node, val) in G.degree()}
    logger.info("Computing degree of each node in %s seconds" % (time.time() - start_time4))
    # Computing betweenness centrality for nodes
    # start_time3 = time.time()
    # beetweennes = nx.betweenness_centrality(G)
    # mean_beetweennes = statistics.mean(list(beetweennes.values()))
    # logger.info("Computing betweenness centrality in %s seconds" % (time.time() - start_time3))
    # Computing the NLC index of each node
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    logger.info("Computing NLC index of each node")
    for i in G:
        neighbors = neighborhood(G, i, level=3)
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(distances[i, j]))
    # sorting in descending order the nodes based on their NLC index  values and degrees vector
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    degrees = dict(sorted(degrees.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    result = dict.fromkeys(nlc_indexes.keys(), 0)
    for node in result:
        result[node] = list(nlc_indexes.keys()).index(node) + list(degrees.keys()).index(node)
    result = dict(sorted(result.items(), key=operator.itemgetter(1), reverse=False))
    k = args.top
    print(list(result.keys())[:k])


def gravity_model_modified(args):
    G = nx.from_scipy_sparse_matrix(load_adjacency_matrix(args.matfile, variable_name=args.matfile_variable_name),
                                    create_using=nx.Graph)
    nodes = G.number_of_nodes()
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.matfile)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing the degree of each node
    degrees = {node: val for (node, val) in G.degree()}
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    ks_max = max(core_values.values())
    ks_min = min(core_values.values())
    attraction_coefficient = dict()
    force_node = dict()
    # K-SHell based on gravity centrality with NetMF embedding for distances
    KSGCNetMF = dict.fromkeys(core_values.keys(), 0)
    for i in G:
        neighbors = neighborhood(G, i, level=3) # Computing 3rd-order neighborhood of node i
        for j in neighbors:
            alias = (i, j)
            attraction_coefficient[alias] = math.exp((core_values[i] - core_values[j]) / (ks_max - ks_min))
            force_node[alias] = attraction_coefficient[alias] * ((degrees[i] * degrees[j]) / distances[alias])
            KSGCNetMF[i] += force_node[alias]
    # sorting in descending order
    nlc_indexes = dict(sorted(KSGCNetMF.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    print(list(KSGCNetMF.keys())[:k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matfile", type=str, required=True,
                        help=".mat input file path of original network")
    parser.add_argument('--matfile-variable-name', default='network',
                        help='variable name of adjacency matrix inside a .mat file.')
    parser.add_argument("--top", default=10, type=int,
                        help="#nodes top nodes")
    parser.add_argument("--output", type=str, required=True,
                        help="top influential nodes output file path")
    parser.add_argument('--algorithm', dest="algorithm", action="store_true",
                        help="using NLC to compute influence of nodes")
    parser.set_defaults(algorithm=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')  # include timestamp
    if args.algorithm:
        logger.info("Algorithm NLC starting")
        start = time.time()
        nlc(args)
        logger.info("Algorithm NLC concluded in %s seconds" % (time.time() - start))
    # else:
    #     logger.info("Algorithm NLC second modified starting")
    #     start = time.time()
    #     nlc_modified_second(args)
    #     logger.info("Algorithm NLC second modified concluded in %s seconds" % (time.time() - start))
    else:
        logger.info("Algorithm LRASP starting")
        start = time.time()
        local_rasp(args)
        logger.info("Algorithm LRASP concluded in %s seconds" % (time.time() - start))
    #else:
    #    logger.info("Algorithm gravity model KSGCNetMF starting")
    #    start = time.time()
    #    gravity_model_modified(args)
    #    logger.info("Algorithm gravity model KSGCNetMF concluded in %s seconds" % (time.time() - start))



