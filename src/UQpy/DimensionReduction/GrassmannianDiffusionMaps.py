from UQpy.DimensionReduction.Kernels import Projection
from UQpy.DimensionReduction.DiffusionMaps import DiffusionMaps
from UQpy.DimensionReduction.Grassmann import Grassmann


class GrassmannianDiffusionMaps(DiffusionMaps):
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

    def __init__(self, alpha=0.5, n_evecs=2, sparse=False, k_neighbors=1, kernel_composition='prod',
                 kernel_object=Projection(), p='max', orthogonal=False):

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
        self.p = p
        self.orthogonal = orthogonal

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

        # Initial checks!
        if not self.orthogonal:
            if kernel_composition in ("left", "right", "prod", "sum"):
                self.kernel_composition = kernel_composition
            elif kernel_composition is None:
                self.kernel_composition = None
            else:
                raise ValueError('UQpy: not valid kernel_composition.')

        super().__init__(alpha=alpha, n_evecs=n_evecs, sparse=sparse, k_neighbors=k_neighbors,
                         kernel_object=kernel_object)

    def fit(self, X=None, kernel_matrix=None, **kwargs):

        """
        Compute the diffusion coordinates.

        In this method, `X` is a list of data points (or matrices/tensors) or an object of ``Grassmann``. The user can
        provide the `kernel_matrix` instead of `X`. If the user wants to use the Gaussian kernel, `epsilon` can be
        provided, otherwise it is computed from the median of the pairwise distances.

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

    def get_kernel_matrix(self, **kwargs):

        #X = self.X

        if isinstance(self.X, Grassmann):
            grassmann_object = self.X

            num_X = len(grassmann_object.X)
            if num_X < 2:
                raise ValueError('UQpy: At least two data points must be provided.')

        elif isinstance(self.X, list):

            num_X = len(self.X)
            if num_X < 2:
                raise ValueError('UQpy: At least two data points must be provided.')

            if self.orthogonal:
                self.kernel_object.fit(X=self.X)
                self.kernel_matrix = self.kernel_object.kernel_matrix

            else:
                grassmann_object = Grassmann(p=self.p)
                grassmann_object.fit(X=self.X)

        else:
            raise TypeError('UQpy: Not valid type for X, it should be either a list or a Grassmann object.')

        if not self.orthogonal:
            if self.kernel_composition in ("prod", "sum"):
                self.kernel_object.fit(X=grassmann_object.psi)
                kernel_left = self.kernel_object.kernel_matrix
                self.kernel_object.fit(X=grassmann_object.phi)
                kernel_right = self.kernel_object.kernel_matrix

                if self.kernel_composition == "prod":
                    self.kernel_matrix = kernel_left * kernel_right
                else:
                    self.kernel_matrix = kernel_left + kernel_right

            elif self.kernel_composition == "left":
                self.kernel_object.fit(X=grassmann_object.psi)
                self.kernel_matrix = self.kernel_object.kernel_matrix

            elif self.kernel_composition == "right":
                self.kernel_object.fit(X=grassmann_object.phi)
                self.kernel_matrix = self.kernel_object.kernel_matrix

            else:
                raise ValueError('UQpy: not valid kernel_composition.')
