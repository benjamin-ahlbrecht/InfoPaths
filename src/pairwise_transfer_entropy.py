import numpy as np

from data import Data
from utilities import transfer_entropy as te


class PairwiseTransferEntropy():
    def __init__(self, X):
        """Deduce causal links from time-series using the pairwise transfer
        entropy.

        Parameters
        ----------
        X : np.ndarray of floats
            Time-series where each row represents a unique observation and each
            column represents a separate time-series feature.
        """
        # Array, where rows/columns correpond to observations/features
        self._X = X

        # Number of observations, number of features
        self._nrows, self._ncols = self._X.shape

        # Takens' embedding time-delays
        self._delays = None

        # Takens' embedding embedding dimension
        self._embeds = None

        # Source -> Destination delay (independent of Takens' embeddings)
        self._sd_delays = None

        # Pairwise transfer entropy array
        self._TE = None

    @property
    def sd_delays(self):
        return self._sd_delays

    @property
    def delays(self):
        return self._delays

    @property
    def embeds(self):
        return self._embeds

    @property
    def pairwise_matrix(self):
        return self._TE

    def compute_delays(self, t_min=1, t_max=5, k=4):
        """Computes the Takens' embedding time delays using delayed mutual
        information.

        Parameters
        ----------
        t_min : int, default=1
            The minimum embedding delay to test.
        t_max : int, default=5
            The maximum embedding delay to test.
        k : int, default=4
            The number of nearest neighbors to sample in the full joint space.

        Returns
        -------
        delays : np.ndarray of ints
            The computed delays for each feature in the data
        """
        # Array to hold a time-delay for each time-series (feature)
        delays = np.zeros(self._ncols, dtype=np.int32)

        # Iterate through each feature and estimate the optimal delay
        for i in range(delays):
            feature = Data(self._X[:, i])
            delays[i] = feature.auto_mi(t_min, t_max, k)

        # Return our estimated delays
        return delays

    def compute_embeds(self, delays=None, d_min=1, d_max=5, cutoff=0.9):
        """Computes the Takens' embedding embedding dimensions using Cao's
        embedding for the minimum embedding dimension.

        Parameters
        ----------
        delay : None, int, or iterable of ints, default=None
            Time-delays to assign to the time-series. If delay=None, values
            will be computed automatically. If a single integer is provided,
            the value will be used for each feature.
        d_min : int, default=1
            The minimum embedding dimension to test
        d_max : int, default=5
            The maximum embedding dimension to test
        cutoff : float, default=0.9
            The cutoff point used to determine the minimum embedding dimension.

        Returns
        -------
        embeds : np.ndarray of ints
        """
        # Compute delays
        if delays is None:
            delays = self.compute_delays()
        elif isinstance(delays, int):
            delays = np.full(self._ncols, delays, dtype=np.int32)

        # Array to hold an embedding-dimension for each time-series (feature)
        embeds = np.zeros(self._ncols, dtype=np.int32)

        # Iterate through each feature and estimate the embedding dimension
        for i in range(embeds):
            feature = Data(self._X[:, i])
            embeds[i] = feature.cao_embed(delays[i], d_min, d_max, cutoff)

        # Return our estimated embedding dimensions
        return embeds

    def pairwise_te(self, sd_delays=1, delays=None, embeds=None, k=4,
                    verbosity=False):
        """Computes the pairwise transfer entropy between a set of time-series
        (features). In the resulting array, rows serve as the source feature,
        while columns serve as the destination feature.

        Parameters
        ----------
        sd_delays : None, int, or 2D np.ndarray of ints, default=1
            The source-destination delays to use for the given source/
            destination where rows serve as the source feature and columns
            serve as the destination feature. If sd_delays=None, the max value
            in the range [1, 5] is used. If an int is used, the corresponding
            value will be used for each soure-destination pair.
        delays : None, int, or np.ndarray of ints, default=None
            The Takens' embedding time-delays to use for each feature. Using
            delays=None will automatically determine the time-delay given a
            pre-supplied/computed value, or it will automatically compute it.
            If a single integer is provied, the value will be used for each
            calculation.
        embeds : None, int, or np.ndarray of ints, default=None
            The Takens' embedding dimension to use for each feature. Using
            embeds=None will automatically determine the embedding dimension
            given a pre-supplied/computed value, or it will automatically
            compute it. If a single integer is provided, the value will be used
            for each calculation.
        k : int, default=4
            The number of nearest neighbors to sample in the full joint space.
        verbosity : bool, default=False
            Whether to provide verbosity when (if) computing source-destination
            delays, Takens' embedding time-delays and embedding dimensions, and
            transfer entropy values.

        Returns
        -------
        TEyx : 2D np.ndarray of floats
            The estimated pairwise transfer entropy.
        """
        # Compute our time-delays and embedding dimensions if necessary
        if delays is None:
            delays = self.compute_delays()
        elif isinstance(delays, int):
            delays = np.full(self._ncols, delays, dtype=np.int32)

        if embeds is None:
            embeds = self.compute_embeds()
        elif isinstance(embeds, int):
            embeds = np.full(self._ncols, embeds, dtype=np.int32)

        # Array to hold pairwise transfer entropy
        TE = np.zeros((self._ncols, self._ncols))

        # Convert our sd_delays to an array
        if isinstance(sd_delays, int):
            sd_delays = np.full((self._ncols, self._ncols), sd_delays)

        # Compute the transfer entropy between each pair of variables
        for i in range(self._ncols):
            for j in range(self._ncols):
                # Automatically compute individual sd_delays
                if sd_delays is None:
                    # Create the sd_delays array
                    sd_delays = np.zeros((self._ncols, self._ncols))

                    # Create value to store delayed TEyx values
                    sd_delay_TEyx = np.zeros(5)

                    # Iterate through each sd_delay in [1, 5], pick the max
                    for k in range(5):
                        sd_delay_TEyx = te(source=self._x[:, i],
                                           destination=self._x[:, j],
                                           delay=k+1,
                                           source_delay=delays[i],
                                           destination_delay=delays[j],
                                           source_embed=embeds[i],
                                           destination_embed=embeds[j],
                                           k=k, locals=False)

                    # Assign the highest value
                    sd_delays[i, j] = np.argmax(sd_delay_TEyx) + 1

                    # Pick the greatest delayed TEyx as the final one
                    TEyx = sd_delay_TEyx[sd_delays[i, j] - 1]

                # Use pre-supplied sd_delay values
                else:
                    TEyx = te(source=self._x[:, i], destination=self._x[:, j],
                              delay=sd_delays[i, j],
                              source_delay=delays[i],
                              destination_delay=delays[j],
                              source_embed=embeds[i],
                              destination_embed=embeds[j],
                              k=k, locals=False)

                # Populate our pairwise array
                TE[i, j] = TEyx

        # Assign calculated values as class attributes for later retrieval
        self._delays = delays
        self._embeds = embeds
        self._sd_delays = sd_delays
        self._TE = TE

        # Return our pairwise transfer entropy array
        return TE
