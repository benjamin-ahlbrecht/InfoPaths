import numpy as np
from scipy.special import psi
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor


def split_arange(start=0, stop=10, batch_size=5):
    splits = []
    while start < stop - batch_size:
        splits.append(range(start, start+batch_size))


        start += batch_size
        if start + batch_size > stop - 1:
            splits.append(range(start, stop))

    return splits


def rnn(args):
    """Radius-neighbors function used to parallize the radius-neighbors search.

    Parameters
    ----------
    args : (cKDTree, x, r, batches, index)
        Function arguments as a tuple, where...
        cKDTree : tuple of cKDTree objects
        x : tuple of np.ndarray of points to sample
        r : tuple of np.ndarray of radii which produced the hyper-square
        batches : tuple of ranges() indicating the indices each batch Computes
        index : tuple of ints representing the batch indices to iterate over

    Returns
    -------
    rnn : np.ndarray of ints
        The radius neighbors for each point queried
    """
    # Unpack the arguments
    kd_tree, x, r, batches, index = args

    # Grab the right batch from our batches
    batch = batches[index]
    n_tasks = len(batch)

    # Fill an empty radius-neighbors array with the queries from the batch
    rnn = np.zeros(n_tasks, dtype=int)

    # Iterate through each task in the batch and assign it to rnn
    for i, task in enumerate(batch):
        rnn[i] = kd_tree.query_ball_point(
            x=x[task], r=r[task], p=np.inf, return_length=True) - 1

    return rnn


def mutual_information(x, y, k=4, batch_size=None):
    """Calculates the mutual information between two time-series using the
    approach proposed by Kraskov, Stoegbauer, and Grassberger (KSG).

    Parameters
    ----------
    x, y : np.ndarray
        Input time-series
    k : int, default=4
        Number of nearest neighbors to sample in the full joint space
    locals : bool, default=False
        Whether to return local values whose expectation produces the mutual
        information
    batch_size : int or None, default=None
        How many radius-neighbors indices each sub-process should handle. If
        batch_size=None, the optimum number will be estimated.

    Returns
    -------
    mi : float or np.ndarray of floats
        The local or expected mutual information value(s)
    """
    # Reshape our data, so it can be inputted properly
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    # Create our joint data
    xy = np.column_stack((x, y))

    # Number of observations
    n = xy.shape[0]

    # Number of tasks each batch should perform
    if batch_size is None:
        batch_size = int(n / 6)

    # Create a batch of tasks to maximize the usage of each sub-process
    batches = split_arange(start=0, stop=n, batch_size=batch_size)
    n_batches = len(batches)

    # Construct the kd-trees
    kd_tree_xy = cKDTree(xy)
    kd_tree_x = cKDTree(x)
    kd_tree_y = cKDTree(y)

    # Generate the hyper-square radius about the joint space
    dists = kd_tree_xy.query(xy, k=k+1, p=np.inf, workers=-1)[0]
    radius = dists[:, -1]

    # Since we only want to query points less than the radius, reduce it a tid
    radius -= 10**-16

    # Create our arguments list for each sub-space
    args_x = ((kd_tree_x, x, radius, batches, i) for i in range(n_batches))
    args_y = ((kd_tree_y, y, radius, batches, i) for i in range(n_batches))

    # Use multiprocessing to estimate the radius-neighbors for each batch
    with ProcessPoolExecutor() as pool:
        results_x = pool.map(rnn, args_x)
        results_y = pool.map(rnn, args_y)

    # Concatenate each batch together into np.ndarrays n_x and n_y
    n_x = []
    n_y = []

    for result in results_x:
        n_x.extend(result)

    for result in results_y:
        n_y.extend(result)

    n_x = np.array(n_x)
    n_y = np.array(n_y)

    # Estimate the mutual information
    mi = psi(n) + psi(k) - np.mean(psi(n_x + 1) + psi(n_y + 1))
    return mi


def conditional_mutual_information(source, destination, condition, k=4, batch_size=None):
    # 1 letter variables will be easier to work with moving on
    x = source
    y = destination
    z = condition

    # Reshape our data, so it can be inputted properly
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    if len(z.shape) == 1:
        z = z.reshape(-1,  1)

    # Define our joint space and the necessary marginal spaces by concatenation
    xyz = np.column_stack((x, y, z))
    xz = np.column_stack((x, z))
    yz = np.column_stack((y, z))

    # Number of observations
    n = xyz.shape[0]

    # Number of tasks each batch should perform
    if batch_size is None:
        batch_size = int(n / 6)

    # Create a batch of tasks to maximize the usage of each sub-process
    batches = split_arange(start=0, stop=n, batch_size=batch_size)
    n_batches = len(batches)

    # Construct the kd-trees
    kd_tree_xyz = cKDTree(xyz)
    kd_tree_xz = cKDTree(xz)
    kd_tree_yz = cKDTree(yz)
    kd_tree_z = cKDTree(z)

    # Generate the hyper-square radius about the joint space
    dists = kd_tree_xyz.query(xyz, k=k+1, p=np.inf, workers=-1)[0]
    radius = dists[:, -1]

    # Since we only want to query points less than the radius, reduce it a tid
    radius -= 10**-16

    # Create our arguments list for each sub-space
    args_xz = ((kd_tree_xz, xz, radius, batches, i) for i in range(n_batches))
    args_yz = ((kd_tree_yz, yz, radius, batches, i) for i in range(n_batches))
    args_z = ((kd_tree_z, z, radius, batches, i) for i in range(n_batches))

    with ProcessPoolExecutor() as executor:
        results_xz = executor.map(rnn, args_xz)
        results_yz = executor.map(rnn, args_yz)
        results_z = executor.map(rnn, args_z)

    # Concatenate each batch together into np.ndarrays n_x and n_y
    n_xz = []
    n_yz = []
    n_z = []

    for result in results_xz:
        n_xz.extend(result)

    for result in results_yz:
        n_yz.extend(result)

    for result in results_z:
        n_z.extend(result)

    n_xz = np.array(n_xz)
    n_yz = np.array(n_yz)
    n_z = np.array(n_z)

    # Estimate the conditional mutual information
    cmi = psi(k) + np.mean(psi(n_z + 1) - psi(n_xz + 1) - psi(n_yz + 1))
    return cmi


def transfer_entropy(source, destination, delay=1, source_delay=1,
                     destination_delay=1, source_embed=1,
                     destination_embed=1, k=4, batch_size=None):
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
        Takens' embedding delay of the source variable.
    destination_delay : int, default=1
        Takens' embedding delay of the destination variable.
    source_embed : int, default=1
        Takens' embedding dimension of the source variable.
    destination_embed : int, default=1
        Takens' embedding dimension of the destination variable.
    k : int, default=4
        The number of nearest neighbors to sample in the full joint space.
    batch_size : int or None, default=None
        How many radius-neighbors indices each sub-process should handle. If
        batch_size=None, the optimum number will be estimated.

    Returns
    -------
    TEyx : float or np.ndarray of floats
        The estimated expected or local value(s) of the transfer entropy from
        the source to the destination.
    """
    # Rename some parameters to reduce clutter
    y = source
    x = destination
    y_delay = source_delay
    x_delay = destination_delay
    y_embed = source_embed
    x_embed = destination_embed

    # Reshape our data, so it can be inputted properly
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    if len(y.shape) == 1:
        x = x.reshape(-1, 1)

    n = y.shape[0]

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

    # Finally, create our Takens' embedding time-series
    xp = np.column_stack([x[xp_cut: xp_cut + xpxy_len]])
    x = np.column_stack([x[x_cut + t * x_delay: x_cut + t * x_delay + xpxy_len]
                        for t in range(x_embed)])
    y = np.column_stack([y[y_cut + t * y_delay: y_cut + t * y_delay + xpxy_len]
                        for t in range(y_embed)])

    # Then, TE(Y -> X) = cmi(xp ; y | x)
    TEyx = conditional_mutual_information(xp, y, x, k=k, batch_size=batch_size)
    return TEyx


def conditional_transfer_entropy(source, destination, conditions,
                                 delay=1, source_delay=1, destination_delay=1,
                                 conditional_delays=1, source_embed=1,
                                 destination_embed=1, conditional_embeds=1,
                                 k=4, batch_size=None):
    """Calculates the transfer entropy from one time-series Y to another time-
    series X using the Kraskov, Stoegbauer, Grassberger (KSG)
    mutual information estimator algorithm 1.

    Parameters
    ----------
    source : np.ndarray
        Array representing the source time-series variable.
    destination : np.ndarray
        Array representing the destination time-series variable.
    conditions : iterable of np.ndarrays
        List where each element represents a source to condition upon.
    delay : int, default=1
        Delay between the source and the destination time series.
    source_delay : int, default=1
        Takens' embedding delay of the source variable.
    destination_delay : int, default=1
        Takens' embedding delay of the destination variable.
    conditional_delays : int or np.ndarray of ints, default=1
        Takens' embedding delay for the conditional variables
    source_embed : int, default=1
        Takens' embedding dimension of the source variable.
    destination_embed : int, default=1
        Takens' embedding dimension of the destination variable.
    conditonal_embeds : int or np.ndarray of ints, default=1
        Takens' embedding dimension for the conditional variables
    k : int, default=4
        The number of nearest neighbors to sample in the full joint space.
    batch_size : int or None, default=None
        How many radius-neighbors indices each sub-process should handle. If
        batch_size=None, the optimum number will be estimated.

    Returns
    -------
    CTEyx : float or np.ndarray of floats
        The estimated expected or local value(s) of the transfer entropy from
        the source to the destination.
    """
    # Rename parameters to reduce clutter
    y = source
    x = destination
    z = conditions
    y_delay = source_delay
    x_delay = destination_delay
    z_delays = conditional_delays
    y_embed = source_embed
    x_embed = destination_embed
    z_embeds = conditional_embeds

    # Reshape our data properly
    if len(source.shape) == 1:
        y = y.reshape(-1, 1)

    if len(destination.shape) == 1:
        x = x.reshape(-1, 1)

    for i, condition in enumerate(z):
        if len(condition.shape) == 1:
            z[i] = condition.reshape(-1, 1)

    n = y.shape[0]

    # Find how far into the future (+) or past (-) each time-series begins
    xp_min = 1
    x_min = -1 * (x_embed - 1) * x_delay
    y_min = 1 - delay - (y_embed - 1) * y_delay
    z_mins = -1 * (z_embeds - 1) * z_delays

    # Select the lowest negative value and add to other time-series
    xpxyz_min = min({xp_min, x_min, y_min, np.min(z_mins)})

    # Calculate how many values we cut from the start of each time_series
    xp_cut = xp_min - xpxyz_min
    x_cut = x_min - xpxyz_min
    y_cut = y_min - xpxyz_min
    z_cuts = z_mins - xpxyz_min

    # Calculate the maximum size so we can enforce a square array
    xpxyz_len = n - max({
        xp_cut,
        x_cut + (x_embed - 1) * x_delay,
        y_cut + (y_embed - 1) * y_delay,
        np.max(z_cuts + (z_embeds - 1) * z_delays)
    })

    # Create our Takens' embedding time-series
    xp = np.column_stack([x[xp_cut: xp_cut + xpxyz_len]])
    x = np.column_stack([x[x_cut + t * x_delay: x_cut + t * x_delay + xpxyz_len]
                        for t in range(x_embed)])
    y = np.column_stack([y[y_cut + t * y_delay: y_cut + t * y_delay + xpxyz_len]
                        for t in range(y_embed)])

    for i, c in enumerate(z):
        z[i] = np.column_stack([c[z_cuts[i] + t * z_delays[i]: z_cuts[i] + t * z_delays[i] + xpxyz_len]
                               for t in range(z_embeds[i])])


    # Condition on Z by simply appending it to X-
    xz = x
    for condition in z:
        xz = np.column_stack((xz, condition))

    # Then, CTE(Y -> X | Z) = cmi(xp ; y | xz)
    CTEyx = conditional_mutual_information(xp, y, xz, k=k, batch_size=batch_size)
    return CTEyx
