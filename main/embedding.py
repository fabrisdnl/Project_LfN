import os
import scipy.io
import scipy.sparse as sparse
from scipy.sparse import csgraph
import networkx as nx
import argparse
import logging

logger = logging.getLogger(__name__)


def load_adjacency_matrix(filename):
    filename = '\\' + filename
    # Gets the current working directory
    current_directory = os.getcwd()
    # Go up one directory from working directory and go in networks directory
    os.chdir("..\\networks")
    # Update the current location
    current_directory = os.getcwd();
    # Get a tuple of all the directories in the folder
    o = [os.path.join(current_directory, o) for o in os.listdir(current_directory)
         if os.path.isdir(os.path.join(current_directory, o))]
    # Search the tuple for the directory you want and open the file
    for item in o:
        if os.path.exists(item + filename):
            file = item + filename
            # Reading the .gml file
            G = nx.read_gml(file)
            # Convert to adjacency matrix
            data = nx.adjacency_matrix(G)

    logger.info("loading gml file %s", file)
    return data


def netmf_large(args):
    logger.info("Running NetMF framework for a large window size...")
    logger.info("Window size is set to be %d", args.window)
    # load adjacency matrix from the .gml file
    A = load_adjacency_matrix(args.input)
    print(A)


def netmf_small(args):
    logger.info("Running NetMF framework for a small window size...")
    logger.info("Window size is set to be %d", args.window)
    # load adjacency matrix from the .gml file
    A = load_adjacency_matrix(args.input)
    print(A)
    # directly compute deepwalk matrix
    #deepwalk_matrix = direct_compute_deepwalk_matrix(A,window=args.window, b=args.negative)

    # factorize deepwalk matrix with SVD
    #deepwalk_embedding = svd_deepwalk_matrix(deepwalk_matrix, dim=args.dim)
    #logger.info("Save embedding to %s", args.output)
    #np.save(args.output, deepwalk_embedding, allow_pickle=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help=".gml input file path")
    parser.add_argument("--output", type=str, required=True,
                        help="embedding output file path")
    parser.add_argument("--rank", default=256, type=int,
                        help="number of eigen-pairs used to approximate normalized graph laplacian.")
    parser.add_argument("--dim", default=128, type=int,
                        help="dimension of embedding")
    parser.add_argument("--window", default=10,
                        type=int, help="context window size")
    parser.add_argument("--negative", default=1.0, type=float,
                        help="negative sampling")
    parser.add_argument('--large', dest="large", action="store_true",
                        help="using netmf framework for large window size")
    parser.add_argument('--small', dest="large", action="store_false",
                        help="using netmf framework for small window size")
    parser.set_defaults(large=True)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

    if args.large:
        netmf_large(args)
    else:
        netmf_small(args)
