import numpy as np
from scipy.special import psi
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor

from time import time


def KSG_transfer_entropy(source, destination, delay=1, source_delay=1,
                         destination_delay=1, source_embed=1,
                         destination_embed=1, k=4, return_locals=False):
    """Calculates the transfer entropy from one time-series Y to another time-
    series X using the Kraskov, Stoegbauer, Grassberger (KSG)
    mutual information estimator algorithm 1.

    Parameters
    ----------
    source : np.ndarray
        Array representing the source time-series variable.
    destination : np.ndarray
        Array representing the destination time-series variable.
    delay : int, default=1
        Delay between the source and the destination time series.
    source_delay : int, default=1
        Taken's embedding delay of the source variable.
    destination_delay : int, default=1
        Taken's embedding delay of the destination variable.
    source_embed : int, default=1
        Taken's embedding dimension of the source variable.
    destination_embed : int, default=1
        Taken's embedding depth of the destination variable.
    k : int, default=None
        The number of nearest neighbors to sample in the full joint space.
    return_locals : bool, default=False
        Whether to return the local values of the estimated transfer entropy
        rather than the expected value.

    Returns
    -------
    TEyx : float or np.ndarray of floats
        The estimated expected or local value(s) of the transfer entropy from
        the source to the destination.
    """
    # Rename parameters to reduce clutter
    y = source
    x = destination
    y_delay = source_delay
    x_delay = destination_delay
    y_embed = source_embed
    x_embed = destination_embed
    n = len(x)

    # Find how far into the future (+) or past (-) each time-series begins
    xp_min = 1
    x_min = -1 * (x_embed - 1) * x_delay
    y_min = 1 - delay - (y_embed - 1) * y_delay

    # Select the lowest negative value and add to other time-series
    xpxy_min = min({xp_min, x_min, y_min})

    # Calculate how many values we cut from the start of each time_series
    xp_cut = xp_min - xpxy_min
    x_cut = x_min - xpxy_min
    y_cut = y_min - xpxy_min

    # Calculate the maximum size so we can enforce a square array
    xpxy_len = n - max({
        xp_cut,
        x_cut + (x_embed - 1) * x_delay,
        y_cut + (y_embed - 1) * y_delay
    })

    # Finally, create our Taken's embedding time-series
    xp = np.array([x[xp_cut: xp_cut + xpxy_len]])
    x = np.array([x[x_cut + t * x_delay: x_cut + t * x_delay + xpxy_len]
                  for t in range(x_embed)])
    y = np.array([y[y_cut + t * y_delay: y_cut + t * y_delay + xpxy_len]
                  for t in range(y_embed)])

    # Concatenate joint and marginal spaces
    xpxy = np.row_stack((xp, x, y)).T
    xpx = np.row_stack((xp, x)).T
    xy = np.row_stack((x, y)).T
    x = x.T

    del y

    # Generate the KD-tree
    kd_tree = NearestNeighbors(
        algorithm="kd_tree", metric="chebyshev", n_neighbors=k)

    # Calculate the hyper-square radius about the full joint space
    kd_tree.fit(xpxy)
    radius = kd_tree.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)
    del xpxy

    # Count the number of neighbors in the necessary marginal spaces
    kd_tree.fit(xpx)
    ind = kd_tree.radius_neighbors(radius=radius, return_distance=False)
    n_xpx = np.array([i.size for i in ind])
    del xpx

    kd_tree.fit(xy)
    ind = kd_tree.radius_neighbors(radius=radius, return_distance=False)
    n_xy = np.array([i.size for i in ind])
    del xy

    kd_tree.fit(x)
    ind = kd_tree.radius_neighbors(radius=radius, return_distance=False)
    n_x = np.array([i.size for i in ind])
    del x

    # Calculate and return the transfer entropy locals
    TEyx = psi(k) + psi(n_x + 1) - psi(n_xy + 1) - psi(n_xpx + 1)

    if return_locals:
        return TEyx

    return TEyx.mean()


def KSG_transfer_entropy_bounds(n, k):
    """Calculates the theoretical maximum and minimum for the KSG transfer
    entropy

    Parameters
    ----------
    n : int
        Number of observations
    k : int
        Number of nearest neighbors
    noise : float
        Amount of gaussian noise to add, referring to the standard deviation

    Returns
    -------
    TEyx_max : float
        The maximum theoretical value of the transfer entropy
    TEyx_min : float
        The minimum theoretical value of the transfer entropy
    """
    # Maximum TE case
    x_max = np.random.uniform(low=0.0, high=1.0, size=n)
    x_max = x_max[:-1]
    y_max = x_max[1:] + np.random.normal(loc=0.0, scale=0.1, size=n - 2)
    TEyx_max = KSG_transfer_entropy(y_max, x_max, k=k)

    # Minimum TE case
    y_min = np.zeros(n - 1) + np.random.uniform(0, high=0.0001, size=n - 1)
    x_min = np.zeros(n - 1) + np.random.uniform(0, high=0.0001, size=n - 1)
    TEyx_min = KSG_transfer_entropy(source=y_min, destination=x_min, k=k)

    return TEyx_max, TEyx_min


def get_var(n, noise=0.1):
    y = np.random.uniform(low=0.0, high=1.0, size=n)
    x = np.zeros_like(y)
    for t in range(1, n):
        x[t] = 0.5 * y[t - 1] + 0.25 * y[t - 2] + \
            np.random.normal(loc=0.0, scale=0.2, size=None)

    return y, x


def main():
    # Import data to analyze
    fname = "1l1m_cap_traj.txt"

    print("Loading data...")
    arr = np.loadtxt(fname)

    # Here, we're gonna remove every 4th value so we don't destroy the computer
    arr2 = []
    for i in range(arr.shape[1]):
        arr2.append([])
        for j in range(arr.shape[0]):
            if j % 4 != 0:
                arr2[i].append(arr[j, i])

    arr = np.array(arr2).T

    np.savetxt("New1l1m.txt", arr)

    nrows, ncols = arr.shape
    dtype = arr.dtype

    print(f"\tnRows = {nrows}\n\tnCols = {ncols}")

    # Calculate normalizing condition for our sample size and k
    print("Estimating transfer entropy bounds...")
    TEyx_max, TEyx_min = KSG_transfer_entropy_bounds(nrows, 4)
    print(f"  TEyx (max) = {round(TEyx_max, 3)}\n"
          + f"  TEyx (min) = {round(TEyx_min, 3)}")

    # Initialize our array to hold pairwise TE
    TEyx_arr = np.zeros((ncols, ncols), dtype=dtype)

    t0 = time()
    print("\n~--+--~ Performing Pairwise Calculations ~--+--~")
    for i in range(ncols):
        t1 = time()
        print(f"Row #{i}... ", end="")

        # Parallize the loop, calculating the array row by row.
        args = [(arr[:, i], arr[:, j]) for j in range(ncols)]
        with ProcessPoolExecutor(max_workers=None) as executor:
            # WARNING: Set max_workers above to avoid a memory leak
            # (Or... downsample your data as I did above)
            results = executor.map(KSG_transfer_entropy, *zip(*args))

        TEyx_arr[i] = np.fromiter(results, count=ncols, dtype=dtype)

        dt1 = time() - t1
        print("Done!" + " " * (5 - len(str(i)))
              + f"({round(100*(i+1)/ncols, 1)}%) ({round(dt1, 3)} sec) "
              + f"({round(dt1/ncols, 3)} sec/call)")

    dt0 = time() - t0
    print("\nTime elapsed:"
          + f"\n  {round(dt0/86400, 2)} days"
          + f"\n   = {round(dt0/3600, 2)} hours"
          + f"\n   = {round(dt0/60, 2)} minutes"
          + f"\n   = {round(dt0/(ncols*ncols), 3)} sec/call")

    # Calculate the normalized KSG TE
    NTEyx_arr = (TEyx_arr - TEyx_min) / (TEyx_max - TEyx_min)

    # Save our TE files
    print("Saving files...")
    np.savetxt("1l1m_ksgTE.txt", TEyx_arr)
    np.savetxt("1l1m_ksgNTE.txt", NTEyx_arr)

    print("Finished!")

    print(TEyx_arr)


if __name__ == "__main__":
    main()
