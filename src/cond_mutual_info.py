import numpy as np
from scipy.special import psi
from sklearn.neighbors import NearestNeighbors


class ConditionalMutualInformation():
    def __init__(self, x, y, condition, k=4):
        """Calculates the conditional mutual information between two time-
        series given a third using the Kraskov, Stoegbauer, Grassberger (KSG)
        mutual information estimator (algorithm 1).

        Arguments
        ---------
        source : np.ndarray
            Array representing the source time-series variable.
        destination : np.ndarray
            Array representing the destination time-series variable.
        condition : np.ndarray
            Array representing the conditonal time-series variable
        """
        # Source and destination time-series
        self.x = x.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

        # Conditional time-series
        self.z = condition.reshape(-1, 1)

        # Number of nearest neighbors to use in the full joint space
        self.k = k

    def set_attributes(self, settings):
        """(re)Set class attributes.

        Arguments
        ---------
        settings : dictionary
            A dictionary containing the desired setting(s) to be changed. The
            following attributes can be used as dictionary keys: 'x', 'y',
            'condition', and 'k'.
        """
        self.x = settings.get('x', self.x)
        self.y = settings.get('y', self.y)
        self.z = settings.get('condition', self.z)
        self.k = settings.get('k', self.k)

    def compute(self, locals=False):
        """Computes the KSG conditional mutual information between a source and
        destination time-series given a third time-series

        Arguments
        ---------
        locals : bool, default=False
            Whether to return local conditonal mutual information values in
            addition to the expected value.

        Returns
            cmi : float
                The estimated conditional mutual information (cmi)
            cmi_locals : np.array or None
                The estimated cmi local values if applicable, else None
        """
        # Define our joint space and the necessary marginal spaces, where
        xyz = np.column_stack((self.x, self.y, self.z))
        xz = np.column_stack((self.x, self.z))
        yz = np.column_stack((self.y, self.z))
        z = self.z

        # Generate the KD-tree
        kd_tree = NearestNeighbors(
            algorithm='kd_tree', metric='chebyshev', n_neighbors=self.k)

        # Calculate the hyper-square radius about the full joint space
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

        # Calculate and return the KSG conditional mutual information
        if locals:
            cmi = psi(self.k) + psi(n_z+1) - psi(n_xz+1) - psi(n_yz+1)
            return cmi.mean(), cmi

        cmi = psi(self.k) + np.mean(psi(n_z+1) - psi(n_xz+1) - psi(n_yz+1))
        return cmi, None
