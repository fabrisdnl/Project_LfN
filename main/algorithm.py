# In this file there will be the main algorithm
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
import argparse
import logging

logger = logging.getLogger(__name__)


def neighborhood(G, node, level):
    return nx.single_source_dijkstra_path_length(G, node, cutoff=level)


def load_adjacency_matrix(file, variable_name="network"):
    os.chdir("..\\datasets")
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


def compute_k_shell_values(G):
    logger.info("Computing K-Shell values of nodes")
    start_time = time.time()
    G.remove_edges_from(nx.selfloop_edges(G))
    k_core = nx.core_number(G)
    print("------- k_core: %s seconds ---------" % (time.time() - start_time))
    return k_core


# algorithm NLC(G,N) in identifying influential spreaders...
def nlc(args):
    G, deepwalk_graph = deepwalk_embedding(args.matfile, variable_name=args.matfile_variable_name)
    core_values = compute_k_shell_values(G)
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    for i in G:
        neighbors = neighborhood(G, i, level=1) # todo: decide which k-level neighborhood
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(- np.linalg.norm(deepwalk_graph[i] - deepwalk_graph[j])))
    # sorting in descending order the nodes based on their NLC index  values
        print(nlc_indexes[i])
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    print(list(nlc_indexes[:k]))


def nlc_modified(args):
    G = nx.from_scipy_sparse_matrix(load_adjacency_matrix(args.matfile, variable_name=args.matfile_variable_name),
                                    create_using=nx.Graph)
    # Using NetMF embedding from karateclub module
    logger.info("NetMF embedding of network in %s dataset", args.matfile)
    start_time = time.time()
    model = ne.NetMF()
    model.fit(G)
    embedding = model.get_embedding()
    print("------- NetMF embedding in %s seconds ---------" % (time.time() - start_time))
    # Computing k-core value of each node
    core_values = compute_k_shell_values(G)
    # Computing the NLC index of each node
    nlc_indexes = dict.fromkeys(core_values.keys(), 0)
    logger.info("Computing NLC index of each node")
    for i in G:
        neighbors = neighborhood(G, i, level=1) # todo: decide which k-level neighborhood
        for j in neighbors:
            nlc_indexes[i] += (core_values[i] * math.exp(- np.linalg.norm(embedding[i] - embedding[j])))
    # sorting in descending order the nodes based on their NLC index  values
    nlc_indexes = dict(sorted(nlc_indexes.items(), key=operator.itemgetter(1), reverse=True))
    # returning top 10 influential nodes
    k = args.top
    print(list(nlc_indexes.keys())[:k])


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
        logger.info("Algorithm NLC concluded in time %s" % (time.time() - start))
    else:
        logger.info("Algorithm NLC modified starting")
        start = time.time()
        nlc_modified(args)  # todo: lettura embedding file .npy
        logger.info("Algorithm NLC modified concluded in %s" % (time.time() - start))


