import scipy.spatial.distance as sd
import sys
from UQpy.Utilities import *


class Distances:
    """
    Similarity: It is a class containing the implementation of the similarity and dissimilarity functions for points in
    several spaces (e.g., Euclidean and Grassmannian).

    The methods of this class is inherited by other classes (``DiffusionMaps``, ``Grassmann``, ``GeometricHarmonics``).

    **Input:**

    * **distance_method** (`callable`)
        Defines the distance metric on the manifold. The user can pass a callable object defining the distance metric
        using two different ways. First, the user can pass one of the methods implemented in the class ``Similarity``.
        Thus, an object containing the attribute `distance_method` is passed as `Similarity.distance_method`. Second,
        the user can pass either a method of a class or a function. On the other hand, if the user implemented a
        function (e.g., `user_distance`) to compute the distance, `distance_method` must assume the following value
        `distance_method = user_distance`, which must be pre-loaded using import. In this regard, the function input
        must contain the first (x0) and second (x1) matrices as arguments (e.g, user_distance(x0,x1)).

    * **kernel_method** (`callable`)
        Object of the kernel function defined on the Grassmann manifold. The user can pass a object using two different
        ways. The user can pass one of the methods implemented in the class ``Similarity``, or an user defined callable
        object as for `distance_method`.

    **Attributes:**

    * **kernel_matrix** (`ndarray`)
        Kernel matrix.

    * **distance_matrix** (`ndarray`)
        Distance matrix.

    * **distance_object** (`callable`)
        Distance method.

    * **kernel_object** (`callable`)
        Kernel method.

    **Methods:**
    """

    def __init__(self):

        self.distance_matrix = None

    def fit(self, X=None, y=None, **kwargs):

        """
        Private method: Estimate the distance/kernel matrix between points in `X`, or the partial matrices for `X` with
        respect to `y`.

        **Input:**

        * **X** (`list`)
            Input data.

        * **y** (`list`)
            Data to compute the partial kernel matrix and/or distance matrix.

        **Output/Returns:**

        * **similarity** (`ndarray`)
            Distance/kernel matrix.
        """

        # Check points for type and shape consistency.
        # -----------------------------------------------------------
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            raise TypeError('UQpy: The input matrices must be either list or numpy.ndarray.')

        nargs = len(X)

        if nargs < 2 and y is None:
            raise ValueError('UQpy: At least two arrays must be provided.')

        # ------------------------------------------------------------

        if y is None:
            # Define the pairs of points to compute the Grassmann distance.

            distance_matrix_ = np.empty((nargs * (nargs - 1)) // 2, dtype=np.double)
            k = 0
            for i in range(0, nargs - 1):
                for j in range(i + 1, nargs):
                    x0 = np.asarray(X[i])
                    x1 = np.asarray(X[j])
                    distance_matrix_[k] = self.distance(x0, x1, **kwargs)
                    k = k + 1

            dissimilarity_diag = []
            for i in range(nargs):
                xd = np.asarray(X[i])
                dissim_diag = self.distance(xd, xd, **kwargs)
                dissimilarity_diag.append(dissim_diag)

            distance_matrix = sd.squareform(distance_matrix_) + np.diag(dissimilarity_diag)

        else:

            nargs_y = len(y)
            distance_matrix = np.empty((nargs, nargs_y), dtype=np.double)

            for i in range(nargs):
                for j in range(nargs_y):
                    x0 = np.asarray(X[i])
                    x1 = np.asarray(y[j])

                    distance_matrix[i, j] = self.distance(x0, x1, **kwargs)

        self.distance_matrix = distance_matrix

    # Default Kernel.
    @staticmethod
    def distance(x0, x1, **kwargs):

        """
        Euclidean distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point.

        * **x1** (`list` or `ndarray`)
            Point.

        **Output/Returns:**

        * **d** (`float`)
            Distance between x0 and x1.

        """

        d = np.linalg.norm(x0 - x1)

        return d


class Euclidean(Distances):

    # Kernels
    @staticmethod
    def distance(x0, x1, **kwargs):
        """
        Euclidean distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point.

        * **x1** (`list` or `ndarray`)
            Point.

        **Output/Returns:**

        * **d** (`float`)
            Distance between x0 and x1.

        """

        d = np.linalg.norm(x0 - x1)

        return d


class ArcLength(Distances):

    @staticmethod
    def distance(x0, x1, **kwargs):

        """
        Grassmann distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Grassmann distance between x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, rank)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        d = np.sqrt(abs(k - l) * np.pi ** 2 / 4 + np.sum(theta ** 2))

        return d


class Chordal(Distances):

    @staticmethod
    def distance(x0, x1, **kwargs):

        """
        Chordal distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Chordal distance between x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r_star = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r_star, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        sin_sq = np.sin(theta) ** 2
        d = np.sqrt(abs(k - l) + np.sum(sin_sq))

        return d


class Procrustes(Distances):

    @staticmethod
    def distance(x0, x1, **kwargs):

        """
        Procrustes distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Procrustes distance between x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))
        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, rank)
        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        sin_sq = np.sin(theta / 2) ** 2
        d = np.sqrt(abs(k - l) + 2 * np.sum(sin_sq))

        return d


class Asimov(Distances):

    @staticmethod
    def projection_distance(x0, x1, **kwargs):

        """
        Projection distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Projection distance between x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))

        if l != k:
            raise NotImplementedError('UQpy: distance not implemented for manifolds with distinct dimensions.')

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, k)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        d = np.max(theta)

        return d


class BinetCauchy(Distances):

    @staticmethod
    def distance(x0, x1, **kwargs):

        """
        Estimate the Binet-Cauchy distance.

        One of the distances defined on the Grassmann manifold is the projection distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Projection distance between x0 and x1.
        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))

        if l != k:
            raise NotImplementedError('UQpy: distance not implemented for manifolds with distinct dimensions.')

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, k)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        cos_sq = np.cos(theta) ** 2
        d = np.sqrt(1 - np.prod(cos_sq))

        return d


class FubiniStudy(Distances):

    @staticmethod
    def distance(x0, x1, **kwargs):

        """
        Estimate the Binet-Cauchy distance.

        One of the distances defined on the Grassmann manifold is the projection distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Projection distance between x0 and x1.
        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))

        if l != k:
            raise NotImplementedError('UQpy: distance not implemented for manifolds with distinct dimensions.')

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, k)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        cos_t = np.cos(theta)
        d = np.arccos(np.prod(cos_t))

        return d

class Martin(Distances):

    @staticmethod
    def distance(x0, x1, **kwargs):

        """
        Estimate the Binet-Cauchy distance.

        One of the distances defined on the Grassmann manifold is the projection distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Projection distance between x0 and x1.
        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))

        if l != k:
            raise NotImplementedError('UQpy: distance not implemented for manifolds with distinct dimensions.')

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, k)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        cos_sq = np.cos(theta) ** 2
        float_min = sys.float_info.min
        index = np.where(cos_sq < float_min)
        cos_sq[index] = float_min
        recp = np.reciprocal(cos_sq)
        d = np.sqrt(np.log(np.prod(recp)))

        return d


class Projection(Distances):

    @staticmethod
    def distance(x0, x1, **kwargs):

        """
        Estimate the Binet-Cauchy distance.

        One of the distances defined on the Grassmann manifold is the projection distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Projection distance between x0 and x1.
        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))

        if l != k:
            raise NotImplementedError('UQpy: distance not implemented for manifolds with distinct dimensions.')

        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, rank)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        d = np.sin(np.max(theta))

        return d


class Spectral(Distances):

    @staticmethod
    def distance(x0, x1, **kwargs):

        """
        Estimate the Binet-Cauchy distance.

        One of the distances defined on the Grassmann manifold is the projection distance.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **distance** (`float`)
            Projection distance between x0 and x1.
        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        l = min(np.shape(x0))
        k = min(np.shape(x1))

        if l != k:
            raise NotImplementedError('UQpy: distance not implemented for manifolds with distinct dimensions.')

        rank = min(l, k)

        r = np.dot(x0.T, x1)
        (ui, si, vi) = svd(r, rank)

        index = np.where(si > 1)
        si[index] = 1.0
        theta = np.arccos(np.diag(si))
        d = 2 * np.sin(np.max(theta) / 2)

        return d
