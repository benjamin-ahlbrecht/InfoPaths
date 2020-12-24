"""Create a pathway network of the various complexes using networkx
"""

from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import sys
import os

sys.path.append("AdumnnnnnnReaseserach/")
import pathway_utilities as pu


def pathways_from_path(path):
    """Calculates the driving and receiving transfer entropy pathways.

    Parameters
    ----------
    path: string
        The absolute path to the TE array file.

    Returns
    -------
    pathways : tuple
        A tuple whos first index contains the driving pathways and the second
        index contains the receiving pathways. Regardless, each pathway is a
        dictionary where the key represents the residue index and the value is
        list of ints.
    """
    mat = np.loadtxt(path)
    direc_mat = pu.calc_direc_mat(mat)
    driving_pathway = {}
    receiving_pathway = {}
    for residue in range(len(direc_mat)):
        driving_pathway[residue] = pu.generate_pathway(
            direc_mat, residue, direction="driving")
        receiving_pathway[residue] = pu.generate_pathway(
            direc_mat, residue, direction="receiving")

    pathways = (driving_pathway, receiving_pathway)
    return pathways

def visualize_pathway(pathways, direction="driving"):
    """Visualizes a driving or receiving pathway

    Parameters
    ----------
    pathway : dictionary
        Dictionary representing a pathway for each residue
    direction : string
        Whether the pathway is "driving" or "receiving"

    Returns
    -------
    G :
    """
    G = nx.DiGraph()
    G.add_nodes_from(pathways.keys())

    # Create our edges and add them to the graph
    directed_edges = {}
    for pathway in pathways.values():
        directed_edges = [(pathway[i], pathway[i+1])
            for i in range(len(pathway) - 1)]
        G.add_edges_from(directed_edges)
    return G


def main():
    # Get the necessary files
    te_dir = [ "AdumnnnnnnReaseserach/TE_matrices/" + fname
        for fname in os.listdir("AdumnnnnnnReaseserach/TE_matrices")]

    for path in te_dir:
        # Generate pathways
        pathways = pathways_from_path(path)
        driving_pathways = pathways[0]
        receiving_pathways = pathways[1]
        G1 = visualize_pathway(driving_pathways, direction='driving')
        G2 = visualize_pathway(receiving_pathways, direction='receiving')

        nx.draw_planar(G1, with_labels=True)
        plt.savefig("test.pdf")




if __name__ == '__main__':
    main()
