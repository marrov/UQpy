import scipy.spatial.distance as sd
from UQpy.Utilities import *


class Kernels:
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

        self.kernel_matrix = None

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

            kernel_matrix_ = np.empty((nargs * (nargs - 1)) // 2, dtype=np.double)
            k = 0
            for i in range(0, nargs - 1):
                for j in range(i + 1, nargs):
                    x0 = np.asarray(X[i])
                    x1 = np.asarray(X[j])
                    kernel_matrix_[k] = self.kernel(x0, x1, **kwargs)
                    k = k + 1

            similarity_diag = []
            for i in range(nargs):
                xd = np.asarray(X[i])
                sim_diag = self.kernel(xd, xd, **kwargs)
                similarity_diag.append(sim_diag)

            kernel_matrix = sd.squareform(kernel_matrix_) + np.diag(similarity_diag)

        else:

            nargs_y = len(y)
            kernel_matrix = np.empty((nargs, nargs_y), dtype=np.double)

            for i in range(nargs):
                for j in range(nargs_y):
                    x0 = np.asarray(X[i])
                    x1 = np.asarray(y[j])

                    kernel_matrix[i, j] = self.kernel(x0, x1, **kwargs)

        self.kernel_matrix = kernel_matrix

        #if self.kernel_object is not None:
        #
        #    if self.kernel_object == self.gaussian_kernel:
        #
        #        distance = self._get_kernel(X=X, y=y, sim_fun=self.euclidean)
        #        if epsilon is None:
        #            # This is a rule of thumb.
        #            epsilon = np.median(distance) ** 2
        #
        #        self.kernel_matrix = np.exp(-np.square(distance) / (4 * epsilon))
        #        self.epsilon = epsilon
        #    else:
        #        sim_fun = self.kernel_object
        #        self.kernel_matrix = self._get_kernel(X=X, y=y, sim_fun=sim_fun)


    # Default Kernel.
    @staticmethod
    def kernel(x0, x1, **kwargs):

        """
        Gaussian kernel.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point.

        * **x1** (`list` or `ndarray`)
            Point.

        **Output/Returns:**

        * **ker** (`float`)
            Kernel value for x0 and x1.

        """
        if 'epsilon' in kwargs.keys():
            epsilon = kwargs['epsilon']
        else:
            raise ValueError('UQpy: epsilon not provided.')

        if epsilon is None:
            raise TypeError('UQpy: epsilon cannot be NoneType.')

        if not isinstance(epsilon, float) and not isinstance(epsilon, int):
            raise TypeError('UQpy: x0 must be either float or int.')

        if epsilon < 0:
            raise ValueError('UQpy: epsilon must be larger than 0.')

        ker = np.exp(-(np.linalg.norm(x0 - x1) ** 2) / (4 * epsilon))

        return ker


class Gaussian(Kernels):

    # Kernels
    @staticmethod
    def kernel(x0, x1, **kwargs):

        """
        Gaussian kernel.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point.

        * **x1** (`list` or `ndarray`)
            Point.

        **Output/Returns:**

        * **ker** (`float`)
            Kernel value for x0 and x1.

        """
        if 'epsilon' in kwargs.keys():
            epsilon = kwargs['epsilon']
        else:
            raise ValueError('UQpy: epsilon not provided.')

        if epsilon is None:
            raise TypeError('UQpy: epsilon cannot be NoneType.')

        if not isinstance(epsilon, float) and not isinstance(epsilon, int):
            raise TypeError('UQpy: x0 must be either float or int.')

        if epsilon < 0:
            raise ValueError('UQpy: epsilon must be larger than 0.')

        ker = np.exp(-(np.linalg.norm(x0 - x1) ** 2) / (4 * epsilon))

        return ker

class Projection(Kernels):

    @staticmethod
    def kernel(x0, x1, **kwargs):

        """
        Projection kernel between x0 and x1.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **ker** (`float`)
            Kernel value for x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        r = np.dot(x0.T, x1)
        ker = np.linalg.norm(r, 'fro') ** 2

        return ker


class BinetCauchy(Kernels):

    @staticmethod
    def kernel(x0, x1, **kwargs):

        """
        Binet-Cauchy kernel between x0 and x1.

        **Input:**

        * **x0** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        * **x1** (`list` or `ndarray`)
            Point on the Grassmann manifold.

        **Output/Returns:**

        * **ker** (`float`)
            Kernel value for x0 and x1.

        """

        if not isinstance(x0, list) and not isinstance(x0, np.ndarray):
            raise TypeError('UQpy: x0 must be either list or numpy.ndarray.')
        else:
            x0 = np.array(x0)

        if not isinstance(x1, list) and not isinstance(x1, np.ndarray):
            raise TypeError('UQpy: x1 must be either list or numpy.ndarray.')
        else:
            x1 = np.array(x1)

        r = np.dot(x0.T, x1)
        ker = np.linalg.det(r) ** 2

        return ker
