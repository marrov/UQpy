from UQpy.Utilities import *
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import scipy.spatial.distance as sd
from UQpy.Utilities import _nn_coord
from UQpy.DimensionReduction.Kernels import Gaussian


class DiffusionMaps:
    """
    Diffusion Maps is a Subclass of Similarity.

    Diffusion maps create a connection between the spectral properties of a diffusion process and the intrinsic geometry
    of datasets. An affinity matrix containing the degree of similarity of data points is either estimated based on the
    euclidean distance, using a Gaussian kernel, or it is computed using any other Kernel definition.

    **Input:**

    * **alpha** (`float`)
        Assumes a value between 0 and 1 and corresponding to different diffusion operators.

    * **n_evecs** (`int`)
        The number of eigenvectors and eigenvalues for the eigendecomposition of the transition matrix.

    * **sparse** (`bool`)
        Is a boolean variable to activate the `sparse` mode for the transition matrix.

    * **k_neighbors** (`int`)
        Used when `sparse` is True to select the k samples close to a given sample in the construction
        of an sparse graph.
        
    * **kernel_object** (`function`)
        A callable object used to compute the kernel matrix. Two different options are provided if no object of
        ``Grassmann`` is provided in ``fit``:

        - kernel_object=Similarity.gaussian_kernel;
        - Using an user defined function as kernel_object=user_kernel;

    **Attributes:**

    * **alpha** (`float`)
        Assumes a value between 0 and 1 and corresponding to different diffusion operators.

    * **n_evecs** (`int`)
        The number of eigenvectors and eigenvalues for the eigendecomposition of the transition matrix.

    * **sparse** (`bool`)
        Is a boolean variable to activate the `sparse` mode for the transition matrix.

    * **k_neighbors** (`int`)
        Used when `sparse` is True to select the k samples close to a given sample in the construction
        of an sparse graph.
    
    * **kernel_matrix** (`ndarray`)
        Kernel matrix.
    
    * **transition_matrix** (`ndarray`)
        Transition kernel of a Markov chain on the data.
        
    * **dcoords** (`ndarray`)
        Diffusion coordinates
    
    * **evecs** (`ndarray`)
        Eigenvectors of the transition kernel of a Markov chanin on the data.
    
    * **evals** (`ndarray`)
        Eigenvalues of the transition kernel of a Markov chanin on the data.

    **Methods:**

    """

    def __init__(self, alpha=0.5, n_evecs=2, sparse=False, k_neighbors=1, kernel_object=Gaussian()):

        self.kernel_object = kernel_object
        self.alpha = alpha
        self.n_evecs = n_evecs
        self.sparse = sparse
        self.k_neighbors = k_neighbors
        self.kernel_matrix = None
        self.transition_matrix = None
        self.X = None
        self.dcoords = None
        self.evecs = None
        self.evals = None

        # Initial checks.
        if alpha < 0 or alpha > 1:
            raise ValueError('UQpy: `alpha` must be a value between 0 and 1.')

        if isinstance(n_evecs, int):
            if n_evecs < 1:
                raise ValueError('UQpy: `n_evecs` must be larger than or equal to one.')
        else:
            raise TypeError('UQpy: `n_evecs` must be integer.')

        if not isinstance(sparse, bool):
            raise TypeError('UQpy: `sparse` must be a boolean variable.')
        elif sparse is True:
            if isinstance(k_neighbors, int):
                if k_neighbors < 1:
                    raise ValueError('UQpy: `k_neighbors` must be larger than or equal to one.')
            else:
                raise TypeError('UQpy: `k_neighbors` must be integer.')

    def fit(self, X=None, kernel_matrix=None, **kwargs):

        """
        Compute the diffusion coordinates.

        In this method, `X` is a list of data points (or matrices/tensors). The user can provide the `kernel_matrix`
        instead of `X`. If the user wants to use the Gaussian kernel, `epsilon` can be
        provided using keyword arguments, otherwise it is computed from the median of the pairwise distances.

        **Input:**

        * **X** (`list`)
            Data points in the ambient space.

        * **kernel_matrix** (`float`)
            Kernel matrix computed by the user.

        * **epsilon** (`float`)
            Parameter of the Gaussian kernel.

        **Output/Returns:**

        """

        # Checks!
        if X is None and kernel_matrix is None:
            raise TypeError('UQpy: either X or kernel_matrix must be provided.')

        if X is not None and kernel_matrix is not None:
            raise TypeError('UQpy: please, provide either X or kernel_matrix.')

        # Construct Kernel Matrix
        if X is not None:
            self.X = X
            if not isinstance(X, list):
                raise TypeError('UQpy: X must be a list.')

            self.get_kernel_matrix(**kwargs)
        elif kernel_matrix is not None:
            self.kernel_matrix = kernel_matrix

        # Estimate the sparse kernel matrix if ``sparse`` is True.
        if self.sparse:
            self.sparse_kernel()

        # Compute transition matrix
        self.get_transition_matrix()

        # Get the diffusion maps.
        self.get_dmaps()

    def get_dmaps(self):

        n = np.shape(self.kernel_matrix)[0]
        if self.n_evecs is None:
            self.n_evecs = n

        n_evecs = self.n_evecs

        # Find the eigenvalues and eigenvectors of ``transition_matrix``.
        if self.sparse:
            evals, evecs = spsl.eigs(self.transition_matrix, k=(n_evecs + 1), which='LR')
        else:
            evals, evecs = np.linalg.eig(self.transition_matrix)

        ix = np.argsort(np.abs(evals))
        ix = ix[::-1]
        s = np.real(evals[ix])
        u = np.real(evecs[:, ix])

        # Truncated eigenvalues and eigenvectors.
        evals = s[:n_evecs]
        evecs = u[:, :n_evecs]

        # Compute the diffusion coordinates.
        dcoords = np.zeros([n, n_evecs])
        for i in range(n_evecs):
            dcoords[:, i] = evals[i] * evecs[:, i]

        # self.transition_matrix = transition_matrix
        self.dcoords = dcoords
        self.evecs = evecs
        self.evals = evals

    def get_kernel_matrix(self, **kwargs):

        num_X = len(self.X)
        if num_X < 2:
            raise ValueError('UQpy: At least two data points must be provided.')

        # Compute the kernel matrix using the Gaussiann kernel.
        self.kernel_object.fit(X=self.X, **kwargs)
        self.kernel_matrix = self.kernel_object.kernel_matrix

    def get_transition_matrix(self):

        if self.kernel_matrix is None:
            raise TypeError('UQpy: kernel_matrix not found!')

        alpha = self.alpha
        # Compute the diagonal matrix D(i,i) = sum(Kernel(i,j)^alpha,j) and its inverse.
        # d, d_inv = self._d_matrix(self.kernel_matrix, self.alpha)
        d = np.array(self.kernel_matrix.sum(axis=1)).flatten()
        d_inv = np.power(d, -alpha)

        # Compute L^alpha = D^(-alpha)*L*D^(-alpha).
        # l_star = self._l_alpha_normalize(self.kernel_matrix, d_inv)
        m = d_inv.shape[0]
        if self.sparse:
            d_alpha = sps.spdiags(d_inv, 0, m, m)
        else:
            d_alpha = np.diag(d_inv)

        l_star = d_alpha.dot(self.kernel_matrix.dot(d_alpha))

        # d_star, d_star_inv = self._d_matrix(l_star, 1.0)
        d_star = np.array(l_star.sum(axis=1)).flatten()
        d_star_inv = np.power(d_star, -1)

        if self.sparse:
            d_star_invd = sps.spdiags(d_star_inv, 0, d_star_inv.shape[0], d_star_inv.shape[0])
        else:
            d_star_invd = np.diag(d_star_inv)

        # Compute the transition matrix.
        self.transition_matrix = d_star_invd.dot(l_star)

    # Private method
    def sparse_kernel(self):

        """
        Private method: Construct a sparse kernel.

        Given the k-nearest neighbors and a kernel matrix, return a sparse kernel matrix.

        **Input:**

        * **kernel_matrix** (`ndarray`)
            Kernel matrix.

        * **k_neighbors** (`float`)
            k-neighbors number used in the construction of the sparse matrix.

        **Output/Returns:**

        * **sparse_kernel_matrix** (`ndarray`)
            Sparse kernel matrix.

        """
        kernel_matrix = self.kernel_matrix
        nrows = np.shape(kernel_matrix)[0]
        for i in range(nrows):
            vec = kernel_matrix[i, :]
            idx = _nn_coord(vec, self.k_neighbors)
            kernel_matrix[i, idx] = 0
            if sum(kernel_matrix[i, :]) <= 0:
                raise ValueError('UQpy: Consider increasing `k_neighbors` to have a connected graph.')

        self.kernel_matrix = sparse_kernel_matrix = sps.csc_matrix(kernel_matrix)

    def parsimonious(self, num_eigenvectors=None):
        """
        This method implements an algorithm to identify the unique eigendirections.

        **Input:**

        * **num_eigenvectors** (`int`):
            An integer for the number of eigenvectors to be tested.

        * **visualization** (`Boolean`):
            Plot a grafic showing the eigenvalues and the corresponding ratios.

        **Output/Returns:**
        * **index** (`ndarray`):
            The indexes of eigenvectors.

        * **residuals** (`ndarray`):
            Residuals used to identify the most parsimonious low-dimensional representation.

        """

        evecs = self.evecs
        n_evecs = self.n_evecs

        if num_eigenvectors is None:
            num_eigenvectors = n_evecs
        elif num_eigenvectors > n_evecs:
            raise ValueError('UQpy: num_eigenvectors cannot be larger than n_evecs.')

        eigvec = np.asarray(evecs)
        eigvec = eigvec[:, 0:num_eigenvectors]

        residuals = np.zeros(num_eigenvectors)
        residuals[0] = np.nan

        # residual 1 for the first eigenvector.
        residuals[1] = 1.0

        # Get the residuals of each eigenvector.
        for i in range(2, num_eigenvectors):
            residuals[i] = self._get_residual(fmat=eigvec[:, 1:i], f=eigvec[:, i])

        # Get the index of the eigenvalues associated with each residual.
        index = np.argsort(residuals)[::-1][:len(self.evals)]

        return index, residuals

    # Private method.
    @staticmethod
    def _get_residual(fmat, f):

        """
        Get the residuals of each eigenvector.

        **Input:**
        * **fmat** (`ndarray`):
            Matrix with eigenvectors for the linear system.

        * **f** (`ndarray`):
            Eigenvector in the right-hand side of the linear system.

        **Output/Returns:**
        * **residuals** (`ndarray`):
            Residuals used to identify the most parsimonious low-dimensional representation
            for a given combination of eigenvectors.

        """

        # Number of samples.
        nsamples = np.shape(fmat)[0]

        # Distance matrix to compute the Gaussian kernel.
        distance_matrix = sd.squareform(sd.pdist(fmat))

        # m=3 is suggested by Nadler et al. 2008.
        m = 3

        # Compute an appropriate value for epsilon.
        # epsilon = np.median(abs(np.square(distance_matrix.flatten())))/m
        epsilon = (np.median(distance_matrix.flatten()) / m) ** 2

        # Gaussian kernel. It is implemented here because of the factor m and the
        # shape of the argument of the exponential is the one suggested by Nadler et al. 2008.
        kernel_matrix = np.exp(-np.square(distance_matrix) / epsilon)

        # Matrix to store the coefficients from the linear system.
        coeffs = np.zeros((nsamples, nsamples))

        vec_1 = np.ones((nsamples, 1))

        for i in range(nsamples):
            # Weighted least squares:

            # Stack arrays in sequence horizontally.
            #        [1| x x x ... x]
            #        [      ...     ]
            # matx = [1| 0 0 0 ... 0]
            #        [      ...     ]
            #        [1| x x x ... x]
            matx = np.hstack([vec_1, fmat - fmat[i, :]])

            # matx.T*Kernel
            matx_k = matx.T * kernel_matrix[i, :]

            # matx.T*Kernel*matx
            wdata = matx_k.dot(matx)
            u, _, _, _ = np.linalg.lstsq(wdata, matx_k, rcond=1e-6)

            coeffs[i, :] = u[0, :]

        estimated_f = coeffs.dot(f)

        # normalized leave-one-out cross-validation error.
        residual = np.sqrt(np.sum(np.square((f - estimated_f))) / np.sum(np.square(f)))

        return residual
