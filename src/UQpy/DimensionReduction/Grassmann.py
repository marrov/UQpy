from Utilities import *


class Grassmann:
    """
    Mathematical analysis on the Grassmann manifold.

    The ``Grassmann`` class contains methods of data analysis on the Grassmann manifold, which is a special case of flag
    manifold. The projection of matrices onto the Grassmann manifold is performed via singular value decomposition(SVD),
    where their dimensionality are reduced. Further, the mapping from the Grassmann manifold to a tangent space
    constructed at a given reference point (logarithmic mapping), as well as the mapping from the tangent space to the
    manifold (exponential mapping) are implemented as methods. Moreover, an interpolation can be performed on the
    tangent space taking advantage of the implemented logarithmic and exponential maps. The Karcher mean can be
    estimated using the methods implemented herein.

    ``Grassmann`` is a Subclass of Similarity.

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

    * **interp_object** (`object`)
        Interpolator to be used in the Tangent space. The user can pass an object defining the interpolator
        via four different ways.

        - Using the ``Grassmann`` method ``linear_interp`` as Grassmann(interp_object=Grassmann.linear_interp).
        - Using an object of ``UQpy.Kriging`` as Grassmann(interp_object=Kriging_Object)
        - Using an object of ``sklearn.gaussian_process`` as Grassmann(interp_object=Sklearn_Object)
        - Using an user defined object (e.g., user_interp). In this case, the function must contain the following
          arguments: `coordinates`, `samples`, and `point`.

    * **karcher_method** (`callable`)
        Optimization method used in the estimation of the Karcher mean. The user can pass a callable object via
        two different ways. First, the user can pass one of the methods implemented in the class ``Grassmann``,
        they are:

        - ``gradient_descent``;
        - ``stochastic_gradient_descent``.

        Second, the user can pass callable objects either as a method of a class or as a function.

    * **kernel_compositions** ('str')
        The method adopted to estimate the operation used to obtain the global kernel. Options: `left` (left singulat
        vectors), `right` (right singular vectors), `prod` (Hadamard's product of the left and right kernels), and
        `sum` (sum of the left and right kernel matrices).

    **Attributes:**

    * **X** (`list`)
        Input dataset.

    * **p** (`int` or `str`)
        Dimension of the p-planes defining the Grassmann manifold G(n,p).

    * **ranks** (`list`)
        Dimension of the embedding dimension for the manifolds G(n,p) of each sample.

    * **samples** (`list` of `list` or `ndarray`)
        Input samples defined as a `list` of matrices.

    * **nargs** (`int`)
        Number of matrices in `samples`.

    * **n_psi** (`int`)
        Ranks of `psi`.

    * **n_phi** (`int`)
        Ranks of `phi`.

    * **max_rank** (`int`)
        Maximum value of `ranks`.

    * **psi** (`list`)
        Left singular eigenvectors from the singular value decomposition of each sample in `samples`
        representing a point on the Grassmann manifold.

    * **sigma** (`list`)
        Singular values from the singular value decomposition of each sample in `samples`.

    * **phi** (`list`)
        Right singular eigenvector from the singular value decomposition of each sample in `samples`
        representing a point on the Grassmann manifold.

    * **skl** (`bool`)
        Boolean variable indicating that scikit-learn is used in the interpolation.

    **Methods:**
    """

    def __init__(self, p=None):

        #if not isinstance(p, int):
        #    raise TypeError('UQpy: p must be integer!')

        #if p < 1:
        #    raise ValueError('UQpy: p must be an integer larger than or equal to 1!')

        self.p = p
        self.X = []
        self.psi = []
        self.sigma = []
        self.phi = []
        self.ranks = []
        self.num_X = 0
        self.max_rank = None

    def fit(self, X=None):
        """
        Set the grassmann manifold and project samples on it via singular value decomposition.

        **Input:**

        * **p** (`int` or `str` or `NoneType`)
            Dimension of the p-planes defining the Grassmann manifold G(n,p). This parameter can assume an integer value
            larger than 0 or the strings `max`, when `p` assumes the maximum rank of the input matrices, or `min` when
            it assumes the minimum one. If `p` is not provided `ranks` will store the ranks of each input
            matrix and each sample will lie on a distinct manifold.

        * **X** (`list`)
            Input samples defined as a `list` of matrices.

        """

        p = self.p
        # Test X for type consistency.
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            raise TypeError('UQpy: `X` must be either a list or numpy.ndarray.')
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        test_0 = all(X_.shape[0] == X[0].shape[0] for X_ in X)
        test_1 = all(X_.shape[1] == X[0].shape[1] for X_ in X)
        if not test_0:
            raise ValueError('UQpy: elements in X must have the same dimension.')

        if not test_1:
            raise ValueError('UQpy: elements in X must have the same dimension.')

        # Get the length of X.
        nargs = len(X)

        # At least one argument must be provided, otherwise show an error message.
        if nargs < 1:
            raise ValueError('UQpy: no data in X.')

        if isinstance(p, str):

            ranks = []
            for i in range(nargs):
                ranks.append(np.linalg.matrix_rank(X[i]))

            if p == "max":
                # Get the maximum rank of the input matrices
                p = int(max(ranks))
            elif p == "min":
                # Get the minimum rank of the input matrices
                p = int(min(ranks))
            else:
                raise ValueError('UQpy: The only allowable input strings are `min` and `max`.')

            ranks = np.ones(nargs) * [int(p)]
            ranks = ranks.tolist()

        else:
            if p is None:
                ranks = []
                for i in range(nargs):
                    ranks.append(np.linalg.matrix_rank(X[i]))
            else:
                if not isinstance(p, int):
                    raise TypeError('UQpy: `p` must be integer.')

                if p < 1:
                    raise ValueError('UQpy: `p` must be an integer larger than or equal to one.')

                for i in range(nargs):
                    if min(np.shape(X[i])) < p:
                        raise ValueError('UQpy: The dimension of the input data is not consistent with `p` of G(n,p).')

                ranks = np.ones(nargs) * [int(p)]
                ranks = ranks.tolist()

        ranks = list(map(int, ranks))

        # Singular value decomposition.
        psi = []  # initialize the left singular eigenvectors as a list.
        sigma = []  # initialize the singular values as a list.
        phi = []  # initialize the right singular eigenvectors as a list.
        for i in range(nargs):
            u, s, v = svd(X[i], int(ranks[i]))
            psi.append(u)
            sigma.append(np.diag(s))
            phi.append(v)

        self.X = X
        self.psi = psi
        self.sigma = sigma
        self.phi = phi
        self.p = p
        self.ranks = ranks
        self.num_X = nargs
        self.max_rank = int(np.max(ranks))

    # ==================================================================================================================
    def append(self, X=None):

        """
        Append samples to the object.

        :return:
        """

        # Test X for type consistency.
        if not isinstance(X, list) and not isinstance(X, np.ndarray):
            raise TypeError('UQpy: `X` must be either a list or numpy.ndarray.')
        elif isinstance(X, np.ndarray):
            X = X.tolist()

        test_0 = all(X_.shape[0] == X[0].shape[0] for X_ in X)
        test_1 = all(X_.shape[1] == X[0].shape[1] for X_ in X)
        if not test_0:
            raise ValueError('UQpy: elements in X must have the same dimension.')

        if not test_1:
            raise ValueError('UQpy: elements in X must have the same dimension.')

        nargs_new = len(X)
        for i in range(nargs_new):
            self.X.append(X[i])

        nargs_final = len(self.X)
        X = self.X
        p = self.p

        ranks = self.ranks
        ranks_new = []
        for i in range(nargs_new):

            if np.min(np.shape(X[i])):
                rnk = int(np.linalg.matrix_rank(X[i]))
                ranks.append(rnk)
                ranks_new.append(rnk)

        if p == "max":
            # Get the maximum rank of the input matrices
            p = int(max(ranks))
        elif p == "min":
            # Get the minimum rank of the input matrices
            p = int(min(ranks))

        ranks = np.ones(nargs_final) * [int(p)]
        ranks = ranks.tolist()

        ranks_new = np.ones(nargs_new) * [int(p)]
        ranks_new = ranks_new.tolist()

        self.ranks = list(map(int, ranks))

        for i in range(nargs_new):
            u, s, v = svd(X[i], int(ranks_new[i]))
            self.psi.append(u)
            self.sigma.append(np.diag(s))
            self.phi.append(v)

    @staticmethod
    def log_map(points_grassmann=None, ref=None):

        """
        Mapping points from the Grassmann manifold on the tangent space.

        It maps the points on the Grassmann manifold, passed to the method using the input argument `points_grassmann`,
        onto the tangent space constructed on ref (a reference point on the Grassmann manifold).
        It is mandatory that the user pass a reference point to the method. Further, the reference point and the points
        in `points_grassmann` must belong to the same manifold.

        **Input:**

        * **points_grassmann** (`list`)
            Matrices (at least 2) corresponding to the points on the Grassmann manifold.

        * **ref** (`list` or `ndarray`)
            A point on the Grassmann manifold used as reference to construct the tangent space.

        **Output/Returns:**

        * **points_tan**: (`list`)
            Point on the tangent space.

        """

        # Show an error message if points_grassmann is not provided.
        if points_grassmann is None:
            raise TypeError('UQpy: No input data is provided.')

        # Show an error message if ref is not provided.
        if ref is None:
            raise TypeError('UQpy: No reference point is provided.')

        # Check points_grassmann for type consistency.
        if not isinstance(points_grassmann, list) and not isinstance(points_grassmann, np.ndarray):
            raise TypeError('UQpy: `points_grassmann` must be either a list or numpy.ndarray.')

        # Get the number of matrices in the set.
        nargs = len(points_grassmann)

        shape_0 = np.shape(points_grassmann[0])
        shape_ref = np.shape(ref)
        p_dim = []
        for i in range(nargs):
            shape = np.shape(points_grassmann[i])
            p_dim.append(min(np.shape(np.array(points_grassmann[i]))))
            if shape != shape_0:
                raise Exception('The input points are in different manifold.')

            if shape != shape_ref:
                raise Exception('The ref and points_grassmann are in different manifolds.')

        p0 = p_dim[0]

        # Check reference for type consistency.
        ref = np.asarray(ref)
        if not isinstance(ref, list):
            ref_list = ref.tolist()
        else:
            ref_list = ref
            ref = np.array(ref)

        # Multiply ref by its transpose.
        refT = ref.T
        m0 = np.dot(ref, refT)

        # Loop over all the input matrices.
        points_tan = []
        for i in range(nargs):
            utrunc = points_grassmann[i][:, 0:p0]

            # If the reference point is one of the given points
            # set the entries to zero.
            if utrunc.tolist() == ref_list:
                points_tan.append(np.zeros(np.shape(ref)))
            else:
                # compute: M = ((I - psi0*psi0')*psi1)*inv(psi0'*psi1)
                minv = np.linalg.inv(np.dot(refT, utrunc))
                m = np.dot(utrunc - np.dot(m0, utrunc), minv)
                ui, si, vi = np.linalg.svd(m, full_matrices=False)  # svd(m, max_rank)
                points_tan.append(np.dot(np.dot(ui, np.diag(np.arctan(si))), vi))

        # Return the points on the tangent space
        return points_tan

    @staticmethod
    def exp_map(points_tangent=None, ref=None):

        """
        Map points on the tangent space onto the Grassmann manifold.

        It maps the points on the tangent space, passed to the method using points_tangent, onto the Grassmann manifold.
        It is mandatory that the user pass a reference point where the tangent space was created.

        **Input:**

        * **points_tangent** (`list`)
            Matrices (at least 2) corresponding to the points on the Grassmann manifold.

        * **ref** (`list` or `ndarray`)
            A point on the Grassmann manifold used as reference to construct the tangent space.

        **Output/Returns:**

        * **points_manifold**: (`list`)
            Point on the tangent space.

        """

        # Show an error message if points_tangent is not provided.
        if points_tangent is None:
            raise TypeError('UQpy: No input data is provided.')

        # Show an error message if ref is not provided.
        if ref is None:
            raise TypeError('UQpy: No reference point is provided.')

        # Test points_tangent for type consistency.
        if not isinstance(points_tangent, list) and not isinstance(points_tangent, np.ndarray):
            raise TypeError('UQpy: `points_tangent` must be either list or numpy.ndarray.')

        # Number of input matrices.
        nargs = len(points_tangent)

        shape_0 = np.shape(points_tangent[0])
        shape_ref = np.shape(ref)
        p_dim = []
        for i in range(nargs):
            shape = np.shape(points_tangent[i])
            p_dim.append(min(np.shape(np.array(points_tangent[i]))))
            if shape != shape_0:
                raise Exception('The input points are in different manifold.')

            if shape != shape_ref:
                raise Exception('The ref and points_grassmann are in different manifolds.')

        p0 = p_dim[0]

        # -----------------------------------------------------------

        ref = np.array(ref)

        # Map the each point back to the manifold.
        points_manifold = []
        for i in range(nargs):
            utrunc = points_tangent[i][:, :p0]
            ui, si, vi = np.linalg.svd(utrunc, full_matrices=False)

            # Exponential mapping.
            x0 = np.dot(np.dot(np.dot(ref, vi.T), np.diag(np.cos(si))) + np.dot(ui, np.diag(np.sin(si))), vi)

            # Test orthogonality.
            xtest = np.dot(x0.T, x0)

            if not np.allclose(xtest, np.identity(np.shape(xtest)[0])):
                x0, unused = np.linalg.qr(x0)  # re-orthonormalizing.

            points_manifold.append(x0)

        return points_manifold
