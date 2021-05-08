import numpy as np
from scipy.special import psi
from sklearn.neighbors import NearestNeighbors
from concurrent.futures import ProcessPoolExecutor


class TransferEntropy():
    def __init__(self, source, destination, **kwargs):
        """Calculates the transfer entropy from one time-series to another
        time-series X using the Kraskov, Stoegbauer, Grassberger (KSG) mutual
        information estimator (algorithm 1).

        Arguments
        ---------
        source : np.ndarray
            Array representing the source time-series variable.
        destination : np.ndarray
            Array representing the destination time-series variable.

        Keyword Arguments
        -----------------
        conditions : None, np.ndarray or iterable of np.ndarrays, default=None
            Array(s) representing additional time-series variable(s) to further
            condition the information transfer.
        k : int, default=4
            The number of nearest neighbors to sample in the full joint space.
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
        """
        # Source time-series; the "information sender"
        self.source = source

        # Destination time-series; the "information receiver"
        self.destination = destination

        # Conditional time-series; removes/adds redundant/synergistic info
        self.conditions = kwargs.get('conditions', 4)

        # Number of nearest neighbors to use in the full joint space
        self.k = kwargs.get('k', 4)

        # Time lag from source to destination time-series
        self.delay = kwargs.get('delay', 1)

        # Time lag within source time-series
        self.source_delay = kwargs.get('source_delay', 1)

        # Time lag with destination time-series
        self.destination_delay = kwargs.get('destination_delay', 1)

        # Embedding dimension of source time-series
        self.source_embed = kwargs.get('source_embed', 1)

        # Embedding dimension of destination time-series
        self.destination_embed = kwargs.get('destination_embed', 1)

    def get_attributes(self):
        """Returns all class attributes as a dictionary
        """
        settings = {
            "source": self.source,
            "destination": self.destination,
            "k": self.k,
            "delay": self.delay,
            "source_delay": self.source_delay,
            "destination_delay": self.destination_delay,
            "source_embed": self.source_embed,
            "destination_embed": self.destination_embed
        }
        return settings

    def set_settings(self, **kwargs):
        """(re)Set class attributes. For definitions, see self.__init()__.
        """
        self.source = kwargs.get('source', self.source)
        self.destination = kwargs.get('destination', self.destination)
        self.k = kwargs.get('k', self.k)
        self.delay = kwargs.get('delay', self.delay)
        self.source_delay = kwargs.get('source_delay', self.source_delay)
        self.destination_delay = kwargs.get(
            'destination_delay', self.destination_delay)
        self.source_embed = kwargs.get('source_embed', self.source_embed)
        self.destination_embed = kwargs.get(
            'destination_embed', self.destination_embed)

    def _embed_data(self):
        """Utilizes Takens' embedding theorem to generate an time-delay
        embedding of the source and destination time-series.

        Returns
        -------
            xp : np.ndarray
                The predicted destination variable,
            x : np.ndarray
                The embedded destination variable,
            y : np.ndarray
                The embedded source variable,
            c : np.ndarray or None
                The embedded conditional time-series if applicable, else None
        """
        pass

    def compute_te(self, locals=False, pval=False, tscore=False):
        """Computes the KSG transfer entropy from the source time-series
        to the destination time-series with an option to return local values,
        a p-values, and an associated tscore. Significance is determined using
        block and Poisson bootstrapping in conjunction.

        Arguments
        ---------
        locals : bool, default=False
            Whether to return the local transfer entropy values.
        pval : bool, default=False
            Whether to return a p-value.
        tscore : bool, default=False
            Whether to return a t-score.
        """
        pass


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
