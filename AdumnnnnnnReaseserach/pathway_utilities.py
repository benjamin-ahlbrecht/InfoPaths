import numpy as np
import os

def get_paths(dir):
    """Returns a list of files with relative paths a given directory
    """
    paths = [f"{dir}/{file}" for file in os.listdir(dir)]
    return paths

def calc_direc_mat(mat):
    """Calculates the directional matrix of an array
    """
    dirc_mat = mat - mat.T
    return dirc_mat

def calc_sum_mat(direc_mat):
    sum_mat = np.sum(direc_mat, axis=0) / len(direc_mat)
    return sum_mat

def calc_arg_sum_mat(sum_mat, order="asc"):
    """argsorts the summed matrix of an array max
    """
    arg_sum_mat = np.argsort(sum_mat)
    return arg_sum_mat

def generate_pathway(direc_mat, res_ind, direction="driving"):
    """Generates a pathway for a given residue.
    """
    pathway = []
    while res_ind not in pathway:
        pathway.append(res_ind)
        if direction is "driving":
            res_ind = direc_mat[res_ind].argmax()
        elif direction is "receiving":
            res_ind = direc_mat[res_ind].argmin()

    pathway.append(res_ind)
    return pathway

def print_pathway(pathway, direction="driving"):
    for index, residue in enumerate(pathway):
        residue += 1
        if direction is "driving":
            if index == 0:
                print(residue, end="")
            elif index == len(pathway) - 1:
                print(f" -> {residue}")
            else:
                print(f" -> {residue}", end="")
        elif direction is "receiving":
            if index == 0:
                print(residue, end="")
            elif index == len(pathway) - 1:
                print(f" <- {residue}")
            else:
                print(f" <- {residue}", end="")
