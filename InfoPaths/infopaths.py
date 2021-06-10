import numpy as np

from time import time
from pickle import dump
from concurrent.futures import ProcessPoolExecutor

from .data import Data
from .utilities import transfer_entropy as te
from .utilities import conditional_transfer_entropy as cte


class InfoPaths():
    def __init__(self, X=None, pairwise=None):
        """Deduce causal links from time-series using the pairwise transfer
        entropy.

        Parameters
        ----------
        X : None or np.ndarray of floats, default=None
            Time-series where each row represents a unique observation and each
            column represents a separate time-series feature.
        pairwise : np.ndarray of
        """
        if not isinstance(X, np.ndarray):
            raise TypeError("Parameter 'X' must be of type np.ndarray")

        if len(X.shape) != 2:
            raise TypeError("Parameter 'X' must be 2-dimensional.")

        if isinstance(X, np.ndarray):
            if len(X.shape) != 2:
                raise TypeError("Parameter 'X' must be 2-dimensional.")

        if not isinstance(pairwise, np.ndarray) and pairwise is not None:
            raise TypeError(
                "Parameter 'pairwise' must be of type np.ndarray or NoneType")

        # Array, where rows/columns correpond to observations/features
        self._X = X

        # Number of observations, number of features
        self._nrows, self._ncols = self._X.shape

        # Time delays array
        self._delays = None

        # Delay embeds array
        self._embeds = None

        # Pairwise transfer entropy array
        self._pairwise_te = None
        if pairwise is not None:
            self._pairwise_te = pairwise

        # Adjacency matrix
        self._adjacency_matrix = None

    @property
    def delays(self):
        return self._delays

    @property
    def embeds(self):
        return self._embeds

    @property
    def pairwise_te(self):
        return self._pairwise_te

    @property
    def adjacency_matrix(self):
        return self._adjacency_matrix

    def save(self, fname):
        """Serializes the InfoPaths() object, so the information can be
        retrieved at a later data.

        Parameters
        ----------
        fname : string
            The filname to serialize the object
        """
        dump(self, open(fname, 'wb'))


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
        for i in range(len(delays)):
            feature = Data(self._X[:, i])
            delays[i] = feature.auto_mi(t_min, t_max, k)

        # Return our estimated delays
        return delays

    def compute_embeds(self, delays=None, d_min=1, d_max=5, cutoff=0.9):
        """Computes the Takens' embedding embedding dimensions using Cao's
        embedding for the minimum embedding dimension.

        Parameters
        ----------
        delay : None, int, or np.ndarray of ints, default=None
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
        elif isinstance(delays, np.ndarray):
            pass
        else:
            raise TypeError("delays must be of type None, int, or np.ndarray.")

        # Array to hold an embedding-dimension for each time-series (feature)
        embeds = np.zeros(self._ncols, dtype=np.int32)

        # Iterate through each feature and estimate the embedding dimension
        for i in range(len(embeds)):
            feature = Data(self._X[:, i])
            embeds[i] = feature.cao_embed(delays[i], d_min, d_max, cutoff)

        # Return our estimated embedding dimensions
        return embeds

    def te_pairwise_parallel(self, args):
        y, x, d, y_d, x_d, y_e, x_e, k, batch_size = args
        return te(y, x, d, y_d, x_d, y_e, x_e, k, batch_size)


    def analyze_destination(self, which, delays, embeds, threshold=0.1, k=4,
                            return_te=False, use_pairwise=False):
        """Uses the pairwise transfer entropy to infer significant sources
        given a destination target.

        Parameters
        ----------
        which : int
            Which destination feature to analyze
        delays : np.ndarray of ints
            The Takens' embedding time-delays to use for each feature.
        embeds : np.ndarray of ints
            The Takens' embedding dimension to use for each feature.
        threshold : float, default=0.1
            A threshold value to determine significant transfer entropy values.
            When a transfer entropy value meets the threshold, it is flagged as
            a potential pathway and further analyis marks it as spurious or
            direct causality.
        k : int, default=4
            The number of nearest neighbors to sample in the full joint space.
        return_te : bool, default=False
            Whether to return the transfer entropy values calculated for the
            destination.
        use_pairwise : bool, default=False
            Whether to use the pairwise matrix provided during class
            instantiation or the most recently computed.

        Returns
        -------
        adjacency : np.ndarray of ints
            An adjacency vector where 1 represents a viable source to the
            selected destination.
        TE : np.ndarray of floats
            Array holding the transfer entropy values from the respective
            sources to the selected destination.
        """
        # List to hold potential transfer entropy sources
        te_sources = []

        # Adjacency vector where 1 represents a viable source
        adjacency = np.zeros(self._ncols)

        # Assign destination variables
        dest = self._X[:, which]
        dest_delay = delays[which]
        dest_embed = delays[which]

        # Load or calculate pairwise transfer entropy
        if use_pairwise:
            # Grab transfer entropy values: sources -> destination
            TE = self._pairwise_te[:, which]
        else:
            # Args used to parallize the calculations
            args = ((self._X[:, i], dest, 1, delays[i], dest_delay, embeds[i], dest_embed, k, None) for i in range(self._ncols))
            with ProcessPoolExecutor() as pool:
                results = pool.map(self.te_pairwise_parallel, args)

            # Array to hold TE values from each source to the target destination
            TE = np.array(list(results))

        # Find potential sources by testing against the threshold
        for i, val in enumerate(TE):
            if val >= threshold:
                te_sources.append(i)

        # If we have no conflicting sources, then the source is causal
        if len(te_sources) == 1:
            adjacency[te_sources[0]] = 1

        # Determine whether each source is direct or spurious causality
        else:
            for i, ind in enumerate(te_sources):
                # Assign source parameters
                source = self._X[:, ind]
                source_delay = delays[ind]
                source_embed = embeds[ind]

                # Grab the indices to create our conditonal array
                condition_inds = te_sources[:i] + te_sources[i + 1:]

                # Construct a list of our conditional features
                conditions = []
                for j, cond_ind in enumerate(condition_inds):
                    conditions.append(self._X[:, cond_ind])

                # Gather our conditional time delays and embeddings
                conditional_delays = np.array([
                    delays[j] for j in condition_inds])
                conditional_embeds = np.array([
                    embeds[j] for j in condition_inds])

                # Estimate the conditional transfer entropy
                CTEyx = cte(
                    source=source, destination=dest, conditions=conditions,
                    delay=1, source_delay=source_delay,
                    destination_delay=dest_delay,
                    conditional_delays=conditional_delays,
                    source_embed=source_embed, destination_embed=dest_embed,
                    conditional_embeds=conditional_embeds, k=k
                )

                # Update adjacency vector if we meet the threshold
                if CTEyx >= threshold:
                    adjacency[ind] = 1

        if return_te:
            return adjacency, TE

        return adjacency

    def analyze_destinations(self, threshold=0.1, delays=None, embeds=None,
                             k=4, verbosity=False, use_pairwise=False):
        """Uses the pairwise transfer entropy to infer significant sources in
        the network.

        Parameters
        ----------
        threshold : float, default=0.1
            A threshold value to determine significant transfer entropy values.
            When a transfer entropy value meets the threshold, it is flagged as
            a potential pathway and further analyis marks it as spurious or
            direct causality.
        delays : None, int, or np.ndarray of ints, default=None
            The Takens' embedding time-delays to use for each feature. Using
            delays=None will automatically determine the time-delay given a
            pre-supplied/computed value, or it will automatically compute it.
        embeds : None, int, or np.ndarray of ints, default=None
            The Takens' embedding dimension to use for each feature. Using
            embeds=None will automatically determine the embedding dimension
            given a pre-supplied/computed value, or it will automatically
            compute it.
        k : int, default=4
            The number of nearest neighbors to sample in the full joint space.
        verbosity : bool, default=False
            Whether to provide verbosity when (if) computing source-destination
            delays, Takens' embedding time-delays and embedding dimensions, and
            transfer entropy values.
        use_pairwise : bool, default=False
            Whether to use the pairwise matrix provided during class
            instantiation or the most recently computed.

        Returns
        -------
        adjacency : np.ndarray of ints
            An adjacency matrix where a value of 1 represents a viable source
            row to the selected destination column.
        """
        # Keep track of our runtime
        if verbosity:
            t0 = time()

        # Create our adjacency matrix
        adjacency = np.zeros((self._ncols, self._ncols))

        # Create our pairwise transfer entropy matrix
        TE = np.zeros((self._ncols, self._ncols))

        # Compute our time-delays
        if delays is None:
            if verbosity:
                print("1. Time Delays: computing... ", end="", flush=True)
            delays = self.compute_delays()
            if verbosity:
                print("Done!")

        elif isinstance(delays, int):
            if verbosity:
                print(f"1. Time Delay: {delays}")
            delays = np.full(self._ncols, delays, dtype=np.int32)

        elif isinstance(delays, np.ndarray):
            if verbosity:
                print("1. Time Delays: supplied!")

        else:
            raise TypeError("delays must be of type None, int, or np.ndarray.")

        # Compute embedding dimensions
        if embeds is None:
            if verbosity:
                print("2. Embedding Dimensions: computing... ", end="", flush=True)
            embeds = self.compute_embeds()
            if verbosity:
                print("Done!")

        elif isinstance(embeds, int):
            if verbosity:
                print(f"2. Embedding Dimension: {embeds}")
            embeds = np.full(self._ncols, embeds, dtype=np.int32)

        elif isinstance(embeds, np.ndarray):
            if verbosity:
                print("2. Embedding Dimensions: supplied!")

        else:
            raise TypeError("embeds must be of type None, int, or np.ndarray.")

        # Populate our arrays
        if verbosity:
            print("3. Inferring Sources:")

        for i in range(self._ncols):
            if verbosity:
                t1 = time()
                print(f"  Destination #{i}... ", end="", flush=True)

            adjacency[i], TE[i] = self.analyze_destination(
                which=i, delays=delays, embeds=embeds, threshold=threshold,
                k=4, return_te=True, use_pairwise=use_pairwise
            )

            if verbosity:
                secs = time() - t1
                mins = secs / 60
                print(
                    f"Done! ~ ({np.round(mins, 2)} min.) ({np.round(secs/self._ncols, 2)} sec./call)")

                # Query the list of sources
                sources = []
                for j, val in enumerate(adjacency[i]):
                    if val == 1:
                        sources.append(j)

                print(f"    Pruned Sources: {sources}")

        if verbosity:
            hours = (time() - t0) / 3600
            print(f"Time Elapsed: ({np.round(hours, 2)} hrs.)")

        # Transpose adjacency matrix so rows -> sources; cols -> destinations
        adjacency = adjacency.T

        # Set class attributes for later retrieval
        self._delays = delays
        self._embeds = embeds
        self._pairwise_te = TE
        self._adjacency_matrix = adjacency

        # Return our adjacency matrix
        return adjacency
