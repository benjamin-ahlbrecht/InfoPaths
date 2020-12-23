import pathway_utilities as pu
import numpy as np

def main():
    dir = "TE_matrices"
    files = pu.get_paths(dir)
    depth = 168

    for file in files:
        print(F"FILE: {file}")

        data_mat = np.loadtxt(file)
        direc_mat = pu.calc_direc_mat(data_mat)
        sum_mat = pu.calc_sum_mat(direc_mat)

        most_driving = pu.calc_arg_sum_mat(sum_mat)
        print(f"\tDRIVING {depth} PATHWAYS")
        for i in range(depth):
            try:
                print(f"\t\t\t#{i+1}\t", end="")
                pathway = pu.generate_pathway(direc_mat, most_driving[i], "driving")
                pu.print_pathway(pathway, direction="driving")
            except IndexError:
                pass

        most_receiving = np.flip(pu.calc_arg_sum_mat(sum_mat))
        print(f"\tRECEIVING {depth} PATHWAYS")
        for i in range(depth):
            try:
                print(f"\t\t#{i+1}\t", end="")
                pathway = pu.generate_pathway(direc_mat, most_receiving[i], "receiving")
                pu.print_pathway(pathway, direction="receiving")
            except IndexError:
                pass




if __name__ == '__main__':
    main()
