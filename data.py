import numpy as np
from utilities import mutual_information as mi


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

    def get_data(self):
        return self.x


    def find_delay(self, method=None):
        """Wrapper function to determine the optimal embedding time-delay in a
        given time-series

        Arguments
        ---------
        method : None or string, default=None
            The method used to identify the time-delay of the data.

        Returns
        -------
        delay : int
            The time-delay associated with the data.
        """
        if method.lower() == "auto_mi":
            return self._auto_mi()

    def find_dimension(self, method=None):
        """Wrapper function to determine the optimal embedding dimension in a
        given time-series

        Arguments
        ---------
        method : None or string, default=None
            The method used to identify the embedding dimension of the data.

        Returns
        -------
        dim : int
            The embedding dimension associated with the data.
        """
        pass

    def embed(self, delay=None, embedding=None):
        """Uses Takens' theorem to embed the data according to a time delay and
        an embedding dimension.

        Arguments
        ---------
        delay : None or int, default=None
            The time delay used to reconstruct the original dynamics of the
            data. If delay=None, then the time delay will be automatically
            assesed using ___.
        embedding : None or int, default=None
            The embedding dimension used to reconstruct the original
            dynamicss of the data. If embedding=None, then the embedding
            dimension will automatically be assesed using ___.
        """
        pass

    def _auto_mi(self, t_min=1, t_max=5, k=4):
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


def main():
    arr = np.random.normal(size=1000)
    data = Data(arr)

    print(data._auto_mi())




if __name__ == '__main__':
    main()
