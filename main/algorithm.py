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

# ---------------------------------------------------------------
#
# Function to load the specified dataset and place it in a
# NetworkX Graph structure, removing self loops and isolated
# nodes from the graph.
# @param: file - the name of the input file
# @param: variable_name - data structure
# @return: G - NetworkX graph of the dataset without self loops
#              and isolated nodes
# ---------------------------------------------------------------
def load_graph(file):
    os.chdir("../datasets")
    if ".mat" in file:
        raw_data = scipy.io.loadmat(file, squeeze_me=True)
        variable_name = "network"
        data = raw_data[variable_name]
        logger.info("loading mat file %s", file)
        G = nx.to_networkx_graph(data, create_using=nx.Graph, multigraph_input=False)
    else:
        logger.info("loading gml file %s", file)
        G = nx.read_gml(file, label="id")
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    G.remove_nodes_from(list(nx.isolates(G)))
    mapping = dict(zip(G, range(0, len(list(G.nodes())))))
    G = nx.relabel_nodes(G, mapping)
    return G


# ---------------------------------------------------------------
#
# Function to load the specified dataset and place it in a
# NetworkX Graph structure.
# @param: file - the name of the input file
# @param: variable_name - data structure
# @return: G - NetworkX graph of the dataset
# ---------------------------------------------------------------
def load_default_graph(file):
    os.chdir("../datasets")
    if ".mat" in file:
        raw_data = scipy.io.loadmat(file, squeeze_me=True)
        variable_name = "network"
        data = raw_data[variable_name]
        logger.info("loading mat file %s", file)
        G = nx.to_networkx_graph(data, create_using=nx.Graph, multigraph_input=False)
    else:
        logger.info("loading gml file %s", file)
        G = nx.read_gml(file, label="id")
    return G

# ---------------------------------------------------------------
#
# Function to find the neighborhood of a node, which can be of
# different level specified by the cutoff, in the graph passed.
# @param: G - the graph
# @param: node - the node of which we have to find neighbors
# @param: level - the level specifying the cutoff on path length
#                 from initial node
# @return: res - all the nodes being in a level-neighborhood (does
#                nor return the initial node in the neighborhood)
# ---------------------------------------------------------------
def neighborhood(G, node, level):
    res = nx.single_source_dijkstra_path_length(G, node, cutoff=level)
    del res[node]
    return res

# does return the initial node as part of the neighborhood
def neighborhood_including_node(G, node, level):
    res = nx.single_source_dijkstra_path_length(G, node, cutoff=level)
    return res


# ---------------------------------------------------------------
#
# Loading graph and its DeepWalk embedding
# @param: file - the name of the input file
# @param: variable_name - data structure
# @return: G, embedding - NetworkX graph of the network in the
#                         input file and its embedding (DeepWalk)
# ---------------------------------------------------------------
def deepwalk_embedding(file):
    G = load_graph(file)
    logger.info("DeepWalk embedding of %s network", file)
    start_time = time.time()
    model = DeepWalk(walk_length=200, dimensions=128, window_size=10)
    model.fit(G)
    embedding = model.get_embedding()
    print("------- %s seconds ---------" % (time.time() - start_time))
    return G, embedding

# ---------------------------------------------------------------
#
# Computing euclidean distance between all embedded nodes
# @param: embedding - embedding of nodes
# @return: distances - structure containing the euclidean
#                      distances between each pair of embedded
#                      nodes
# ---------------------------------------------------------------
def precomputing_euclidean(embedding):
    start = time.time()
    distances = dict()
    for i in range(len(embedding)):
        for j in range(len(embedding)):
            alias = (i,j)
            distances[alias] = -np.linalg.norm(embedding[i]-embedding[j])
    logger.info("Precomputing euclidean distances between node embeddings in %s seconds" % (time.time()-start))
    return distances

# ---------------------------------------------------------------
#
# Compute the core number (the largest value k of a k-core
# containing that node) of each node.
# @param: G - NetworkX graph
# @return: k_core - dictionary keyed by node which values are the
#                   core numbers
# ---------------------------------------------------------------
def compute_k_core_values(G):
    start_time = time.time()
    k_core = nx.core_number(G)
    logger.info("Computed K-Core values in %s seconds" % (time.time() - start_time))
    return k_core

# ---------------------------------------------------------------
#
# An other method to compute k_shell values.
# ---------------------------------------------------------------

def check(h, d):
    f = 0
    for i in h.nodes():
        if (h.degree(i) <= d):
            f = 1
            break
    return f


def find_nodes(h, it):
    s = []
    for i in h.nodes():
        if (h.degree(i) <= it):
            s.append(i)
    return s


def kShell_values(h):
    start = time.time()
    it = 1
    tmp = []
    buckets = []

    while (1):
        flag = check(h, it)
        if (flag == 0):
            it += 1
            buckets.append(tmp)
            tmp = []
        if (flag == 1):
            node_set = find_nodes(h, it)
            for each in node_set:
                h.remove_node(each)
                tmp.append(each)
        if (h.number_of_nodes() == 0):
            buckets.append(tmp)
            break

    core_values = dict()

    value = 1

    for b in buckets:
        for n in b:
            core_values[n] = value
        value += 1
    logger.info(("kShell in %s seconds") % (time.time()-start))
    return core_values


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

# ---------------------------------------------------------------
#
# Algorithms for finding ASP for a graph H
# @param: H - graph
# @return: asp_value - average shortest path of graph H
# ---------------------------------------------------------------

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

# ---------------------------------------------------------------
#
# LRASP (Local Relative change of Average Shortest Path
# Centrality) algorithm to obtain the k nodes with higher
# LRASP measures.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def local_rasp(args):
    G = load_default_graph(args.input)
    lrasp = dict.fromkeys(list(G.nodes), 0)
    for i in G:
        neighbors = neighborhood_including_node(G, i, level=2)
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


# ---------------------------------------------------------------
#
# NLC index algorithm to obtain the influence of nodes
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc(args):
    G, deepwalk_graph = deepwalk_embedding(args.input)
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    core_values = compute_k_core_values(G)
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    for i in G:
        neighbors = neighborhood(G, i, level=3)
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(- np.linalg.norm(deepwalk_graph[i] - deepwalk_graph[j])))
    # sorting in descending order the nodes based on their NLC index  values
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    print(list(nlc_indexes.keys())[:k])


# ---------------------------------------------------------------
#
# NLC index algorithm to obtain the influence of nodes
# but with NetMF embedding
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc2(args):
    G = load_graph(args.inpu)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    for i in G:
        neighbors = neighborhood(G, i, level=3)
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(distances[i,j]))
    # sorting in descending order the nodes based on their NLC index  values
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    print(list(nlc_indexes.keys())[:k])


# ---------------------------------------------------------------
#
# NLC modified algorithm to obtain the influence of nodes,
# considering the mean degree of nodes.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc_modified(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # Computing degrees for nodes
    start_time2 = time.time()
    degrees = {node: deg for (node, deg) in G.degree()}
    mean_degree = statistics.mean(list(degrees.values()))
    logger.info("Computing degree centrality in %s seconds" % (time.time() - start_time2))
    # Computing the NLC index of each node
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    logger.info("Computing NLC index of each node")
    for i in G:
        if degrees[i] > mean_degree:
            neighbors = neighborhood(G, i, level=3)
            for j in neighbors:
                nlc_indexes[i] += (core_values[i] * math.exp(distances[i,j]))
    # sorting in descending order the nodes based on their NLC index  values
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    print(list(nlc_indexes.keys())[:k])

# ---------------------------------------------------------------
#
# NLC modified second algorithm to obtain the influence of nodes,
# considering the degrees of nodes.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc_modified_second(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # Computing degree for each node
    start_time4 = time.time()
    degrees = {node: val for (node, val) in G.degree()}
    logger.info("Computing degree of each node in %s seconds" % (time.time() - start_time4))
    # Computing the NLC index of each node
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    logger.info("Computing NLC index of each node")
    for i in G:
        neighbors = neighborhood(G, i, level=3)
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(distances[i,j]))
    # returning top 10 influential nodes
    result = dict.fromkeys(nlc_indexes.keys(), 0)
    for node in result:
        result[node] = nlc_indexes[node] * degrees[node]
    result = dict(sorted(result.items(), key=operator.itemgetter(1), reverse=False))
    k = args.top
    print(list(result.keys())[:k])


# ---------------------------------------------------------------
#
# NLC modified third algorithm to obtain the influence of nodes,
# considering the ratio between the node's degree and the total
# degree of neighborhood of each node.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc_modified_third(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # Computing degree for each node
    start_time4 = time.time()
    degrees = {node: val for (node, val) in G.degree()}
    logger.info("Computing degree of each node in %s seconds" % (time.time() - start_time4))
    # Computing the NLC index of each node
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    neighborhood_degree = dict.copy(nlc_indexes)
    influence = dict.copy(nlc_indexes)
    logger.info("Computing NLC index and degree ratio of each node")
    for i in G:
        neighbors = neighborhood(G, i, level=3)
        neighborhood_degree[i] += degrees[i]
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(distances[i, j]))
            neighborhood_degree[i] += degrees[j]
        influence[i] = degrees[i] / neighborhood_degree[i]
    # returning top 10 influential nodes
    result = dict.fromkeys(nlc_indexes.keys(), 0)
    for node in result:
        result[node] = nlc_indexes[node] * influence[node]
    result = dict(sorted(result.items(), key=operator.itemgetter(1), reverse=False))
    k = args.top
    print(list(result.keys())[:k])

# ---------------------------------------------------------------
#
# KDEC algorithm to obtain the k top nodes in the
# ranking obtained.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def kdec(args):
    G = load_graph(args.input)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # Computing degrees
    degrees = {node: deg for (node, deg) in G.degree()}
    # Computing weights
    weights = {key: core_values[key] * degrees[key] for key in core_values}
    # KDEC structure
    kdec = dict.fromkeys(core_values.keys(), 0.0);
    for i in G:
        neighbors = neighborhood(G, i, level=1)
        for j in neighbors:
            eff_ij = (1 - math.log(1 / degrees[j]))
            kdec[i] += ((weights[i] * weights[j]) / (math.pow(eff_ij, 2)))
    # sorting in descending order the nodes based on their values
    kdec = dict(sorted(kdec.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    print(list(kdec.keys())[:k])

# ---------------------------------------------------------------
#
# KDEC algorithm version which considers the embedding nodes too.
# @param: args - arguments from command line
# Prints the top K influential nodes in the graph
# ---------------------------------------------------------------
def nlc_kdec(args):
    G = load_graph(args.input)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.input)
    start_time1 = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
    G.remove_edges_from(list(nx.selfloop_edges(G)))
    # Precomputing euclidean distances between nodes in embedding
    distances = precomputing_euclidean(embedding)
    # Computing k-core value of each node
    core_values = compute_k_core_values(G)
    # Computing degrees
    degrees = {node: deg for (node, deg) in G.degree()}
    # Computing weights
    weights = {key: core_values[key] * degrees[key] for key in core_values}
    # print(weights)
    # KDEC structure
    nlckdec = dict.fromkeys(core_values.keys(), 0.0)
    for i in G:
        neighbors = neighborhood(G, i, level=1)
        for j in neighbors:
            eff_ij = (1 - math.log(1 / degrees[j]))
            nlckdec[i] += ((weights[i] * weights[j]) / (math.pow(eff_ij, 2))) * math.exp(distances[i,j])

    # sorting in descending order the nodes based on their values
    nlckdec = dict(sorted(nlckdec.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    print(list(nlckdec.keys())[:k])


# def gravity_model_modified(args):
#     G = load_graph(args.input)
#     nodes = G.number_of_nodes()
#     # Using NetMF embedding from karateclub module
#     logger.info("NetMF embedding of network in %s dataset", args.input)
#     start_time1 = time.time()
#     model = ne.NetMF()
#     model.fit(G)
#     embedding = model.get_embedding()
#     logger.info("NetMF embedding in %s seconds" % (time.time() - start_time1))
#     G.remove_edges_from(list(nx.selfloop_edges(G)))
#     # Precomputing euclidean distances between nodes in embedding
#     distances = precomputing_euclidean(embedding)
#     # Computing the degree of each node
#     degrees = {node: val for (node, val) in G.degree()}
#     # Computing k-core value of each node
#     core_values = compute_k_core_values(G)
#     ks_max = max(core_values.values())
#     ks_min = min(core_values.values())
#     attraction_coefficient = dict()
#     force_node = dict()
#     # K-SHell based on gravity centrality with NetMF embedding for distances
#     KSGCNetMF = dict.fromkeys(core_values.keys(), 0)
#     for i in G:
#         neighbors = neighborhood(G, i, level=3) # Computing 3rd-order neighborhood of node i
#         for j in neighbors:
#             alias = (i, j)
#             attraction_coefficient[alias] = math.exp((core_values[i] - core_values[j]) / (ks_max - ks_min))
#             force_node[alias] = attraction_coefficient[alias] * ((degrees[i] * degrees[j]) / distances[alias])
#             KSGCNetMF[i] += force_node[alias]
#     # sorting in descending order
#     nlc_indexes = dict(sorted(KSGCNetMF.items(), key=operator.itemgetter(1), reverse=True))
#     # returning top 10 influential nodes
#     k = args.top
#     print(list(KSGCNetMF.keys())[:k])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help=".mat input file path of original network")
    parser.add_argument("--top", default=10, type=int,
                        help="#nodes top nodes")
    parser.add_argument('--nlc', dest="nlc", action="store_true",
                        help="using NLC to compute influence of nodes")
    parser.set_defaults(nlc=False)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s')  # include timestamp
    if args.nlc:
        logger.info("Algorithm NLC starting")
        start = time.time()
        nlc(args)
        logger.info("Algorithm NLC concluded in %s seconds" % (time.time() - start))
    # else:
    #     logger.info("Algorithm NLC modified starting")
    #     start = time.time()
    #     nlc_modified(args)
    #     logger.info("Algorithm NLC modified concluded in %s seconds" % (time.time() - start))
    else:
        logger.info("Algorithm NLC second modified starting")
        start = time.time()
        nlc_modified_second(args)
        logger.info("Algorithm NLC second modified concluded in %s seconds" % (time.time() - start))
    # else:
    #     logger.info("Algorithm NLC third modified starting")
    #     start = time.time()
    #     nlc_modified_third(args)
    #     logger.info("Algorithm NLC third modified concluded in %s seconds" % (time.time() - start))
    # else:
    #      logger.info("Algorithm LRASP starting")
    #      start = time.time()
    #      local_rasp(args)
    #      logger.info("Algorithm LRASP concluded in %s seconds" % (time.time() - start))
    #else:
    #    logger.info("Algorithm gravity model KSGCNetMF starting")
    #    start = time.time()
    #    gravity_model_modified(args)
    #    logger.info("Algorithm gravity model KSGCNetMF concluded in %s seconds" % (time.time() - start))
    # else:
    #     logger.info("Algorithm KDEC starting")
    #     start = time.time()
    #     kdec(args)
    #     logger.info("Algorithm KDEC concluded in %s seconds" % (time.time() - start))
    # else:
    #     logger.info("Algorithm NLC KDEC starting")
    #     start = time.time()
    #     nlc_kdec(args)
    #     logger.info("Algorithm NLC KDEC concluded in %s seconds" % (time.time() - start))


