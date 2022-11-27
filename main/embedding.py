import os
import sys
import networkx as nx
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("I'm running main")
    # Reading filename from command line
    filename = '\\' + sys.argv[1]
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
            # Displaying nodes
            G.nodes
