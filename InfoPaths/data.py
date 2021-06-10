import numpy as np
from .utilities import mutual_information as mi
from sklearn.neighbors import NearestNeighbors


class Data():
    def __init__(self, x):
        """Class used to preprocess time-series data.

        Arguments
        ---------
        x : np.ndarray
            Numpy array representing time-series data increasing as a discrete
            function of time.
        """
        # We'll just use the self.set_data() method to set our attribute self.x
        self.set_data(x)

    def set_data(self, x):
        ndims = len(x.shape)
        if ndims == 1:
            self.x = x.reshape(-1, 1)
        elif ndims == 2:
            self.x = x
        else:
            raise ValueError("Input array must be 1- or 2-dimensional.")

    def reconstruct(self, delay=None, embed=None):
        """Uses Takens' theorem to embed the data according to a time delay and
        an embedding dimension.

        Arguments
        ---------
        delay : None or int, default=None
            The time delay used to reconstruct the original dynamics of the
            data. If delay=None, then the time delay will be automatically
            assesed using delayed mutual information ('auto_mi').
        embed : None or int, default=None
            The embedding dimension used to reconstruct the original
            dynamicss of the data. If embedding=None, then the embedding
            dimension will automatically be assesed using Cao's embedding
            method ('cao').

        Returns
        -------
        data_embed : np.ndarray
            The reconstructed time-series produced given a time-delay and an
            embedding dimension.
        """
        # Automatically find the embedding dimension/time delay if necessary
        if delay is None:
            delay = self.auto_mi()
        if embed is None:
            embed = self.cao_embed(delay)

        # We must splice the ends of the arrays so they are not jagged
        max_len = self.x.shape[0] - (embed-1)*delay
        if max_len < 1:
            raise ValueError("Time delay and embedding are too large.")

        # Generate and return the delay-embedded time-series
        data_embed = np.column_stack([
            self.x[d*delay: d*delay + max_len] for d in range(embed)
        ])

        return data_embed

    def auto_mi(self, t_min=1, t_max=5, k=4):
        """Uses the first minimum of consecutive delayed mutual informations to
        estimate the optimal time delay.

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
        t : int
            The estimated embedding delay for the data.
        """
        # Dictionary to store MI delays and corresponding values
        mi_vals = {}
        mi_max = -np.inf

        # Iterate through each delay, calculate mi, and test for minima
        for t in range(t_min, t_max+1):
            # Estimate the [delayed] mi
            mi_vals[t] = mi(self.x[t:], self.x[:-t], k=k)

            # Test if the delayed mi lies within a local minima
            if (t > t_min+1) and mi_vals[t-1] <= min(mi_vals[t-2], mi_vals[t]):
                return t-1

            # Update maximum value
            if mi_vals[t] > mi_max:
                mi_max = mi_vals[t]

        # IF the mi is monotonically increasing, t=t_min
        if mi_vals[t_max] == mi_max:
            return t_min

        # IF the mi is monotonically decreasing, t=t_max
        return t_max

    def cao_embed(self, delay, d_min=1, d_max=5, cutoff=0.9):
        """Uses Cao's embedding method to determine the minimum embedding
        dimension. This method serves as a parameter-free alternative to the
        more common method of false nearest neighbors.

        Parameters
        ----------
        delay : int
            A time-delay to assign to the time-series
        d_min : int, default=1
            The minimum embedding dimension to test
        d_max : int, default=5
            The maximum embedding dimension to test
        cutoff : float, default=0.9
            The cutoff point used to determine the minimum embedding dimension.

        Returns
        -------
        embed : int
            The estimated embedding dimension of the time-series
        """
        # TODO: Determine E2(d) to distinguish determinstic/stochastic signals
        # List to store knn distances
        dists = [0, 0, 0]

        # Calculate Cao's E1(d) by sequentially embedding the time-series
        for d in range(d_min, d_max+3):
            # Index to correctly store/retrieve values in dists list
            index = (d - d_min) % 3

            # Find nearest neighbors for dimension d
            x_d = self.reconstruct(delay, embed=d)
            knn_d = NearestNeighbors(n_neighbors=2, metric='chebyshev')
            knn_d.fit(x_d)

            # Query neighbors
            dists_d = knn_d.kneighbors(X=x_d)[0][:, 1]

            # Assign distances to an index in our dists lists
            dists[index] = dists_d

            # Calculate E1(d) = E(d+1) / E(d)
            if (d - d_min) > 1:
                # Get the distances for the respective dimensions
                dists_d = dists[index-2]
                dists_d1 = dists[index-1]
                dists_d2 = dists[index]

                # Calculate E(d) and E(d+1)
                e_d = (dists_d1 / dists_d[:-delay]).mean()
                e_d1 = (dists_d2 / dists_d1[:-delay]).mean()

                # Calculate E1(d)
                e1_d = e_d1 / e_d

                # Return the embedding dimension if it surpasses our threshold
                if e1_d > cutoff:
                    return d-2

        # Return d_max as our best guess to the minimum embedding dimension
        return d_max
