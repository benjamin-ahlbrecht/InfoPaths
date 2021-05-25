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
        Number of nearest neighbors to sample in the full joint space
    locals : bool, default=False
        Whether to return local values whose expectation produces the mutual
        information

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

    # Save some cpu cycles by avoiding superfluous addition for expected value
    if locals:
        # Calculate and return local mutual information values
        mi = psi(n) + psi(k) - psi(n_x+1) - psi(n_y+1)
        return mi

    # Calculate and return expected mutual information values
    mi = psi(n) + psi(k) - np.mean(psi(n_x+1) + psi(n_y+1))
    return mi


def conditional_mutual_information(source, destination, condition, k=4,
                                   locals=False):
    """Calculates the conditional mutual information between two time-
    series given a third using the Kraskov, Stoegbauer, Grassberger (KSG)
    mutual information estimator (algorithm 1).

    Parameters
    ----------
    source : np.ndarray
        Array representing the source time-series variable.
    destination : np.ndarray
        Array representing the destination time-series variable.
    condition : np.ndarray
        Array representing the conditonal time-series variable
    k : int, default=4
        Number of nearest neighbors to sample in the full joint space.
    locals : bool, default=False
        Whether to return local values whose expectation produces the
        conditional mutual information

    Returns
    -------
    cmi : float or np.ndarray of floats
    """
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

    # Generate the KD-tree to create joint radii and sample marginal neighbors
    kd_tree = NearestNeighbors(
        algorithm='kd_tree', metric='chebyshev', n_neighbors=k)

    # Calculate the hyper-square radius about the full joint space using k
    kd_tree.fit(xyz)
    radius = kd_tree.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)
    del xyz

    # Count the number of neighbors in the necessary marginal spaces
    kd_tree.fit(xz)
    ind = kd_tree.radius_neighbors(radius=radius, return_distance=False)
    n_xz = np.array([i.size for i in ind])
    del xz

    kd_tree.fit(yz)
    ind = kd_tree.radius_neighbors(radius=radius, return_distance=False)
    n_yz = np.array([i.size for i in ind])
    del yz

    kd_tree.fit(z)
    ind = kd_tree.radius_neighbors(radius=radius, return_distance=False)
    n_z = np.array([i.size for i in ind])
    del z

    # We save some cpu cycles by avoiding superfluous addition at times
    if locals:
        # Calculate and return local conditional mutual information values
        cmi = psi(k) + psi(n_z+1) - psi(n_xz+1) - psi(n_yz+1)
        return cmi

    # Calculate and return expected conditional mutual information values
    cmi = psi(k) + np.mean(psi(n_z+1) - psi(n_xz+1) - psi(n_yz+1))
    return cmi


def transfer_entropy(source, destination, delay=1, source_delay=1,
                     destination_delay=1, source_embed=1,
                     destination_embed=1, k=4, locals=False):
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
    locals : bool, default=False
        Whether to return the local values of the estimated transfer entropy
        rather than the expected value.

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
    TEyx = conditional_mutual_information(xp, y, x, k=k, locals=locals)
    return TEyx


def conditional_transfer_entropy(source, destination, conditions,
                                 delay=1, source_delay=1, destination_delay=1,
                                 conditional_delays=1, source_embed=1,
                                 destination_embed=1, conditional_embeds=1,
                                 k=4, locals=False):
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
    locals : bool, default=False
        Whether to return the local values of the estimated transfer entropy
        rather than the expected value.

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
    CTEyx = conditional_mutual_information(xp, y, xz, k=k, locals=locals)
    return CTEyx
