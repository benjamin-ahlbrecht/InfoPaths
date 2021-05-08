import numpy as np
from scipy.special import psi
from sklearn.neighbors import NearestNeighbors


def mutual_information(x, y, k=4, locals=False):
    """Calculates the mutual information between two time-series using the
    approach proposed by Kraskov, Stoegbauer, and Grassberger (KSG).

    Parameters
    ----------
    x, y : np.ndarray
        Input time-series
    k : int, default=4
        Number of nearest neighbors to sample in the full joint space.
    locals : bool, default=False
        Whether to return local values in addition to the expected value of the
        mutual information
    """
    # Joint data
    xy = np.column_stack((x, y))

    # Number of observations
    n = xy.shape[0]

    # Generate the KD-tree
    kd_tree = NearestNeighbors(
        algorithm='kd_tree', metric='chebyshev', n_neighbors=k)

    # Calculate the hyper-square radius about the full joint space
    kd_tree.fit(xy)
    radius = kd_tree.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)
    del xy

    # Count the number of neighbors in the necessary marginal spaces
    kd_tree.fit(x)
    ind = kd_tree.radius_neighbors(radius=radius, return_distance=False)
    n_x = np.array([i.size for i in ind])
    del x

    kd_tree.fit(y)
    ind = kd_tree.radius_neighbors(radius=radius, return_distance=False)
    n_y = np.array([i.size for i in ind])
    del y

    if locals:
        mi = psi(n) + psi(k) - psi(n_x+1) - psi(n_y+1)
        return mi.mean(), mi

    mi = psi(n) + psi(k) - np.mean(psi(n_x+1) + psi(n_y+1))
    return mi
