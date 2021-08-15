import copy
from UQpy.Utilities import *
from UQpy.DimensionReduction.Grassmann import Grassmann
from UQpy.DimensionReduction.Distances import ArcLength


class KarcherMean(Grassmann):

    def __init__(self, distance_object=ArcLength()):

        self.X = None
        self.distance_object = distance_object
        self.p = None
        self.karcher_mean = None

        # Instantiating Similarity.
        super().__init__(p=None)

    def fit(self, X=None, append=False):

        # X are points on the Grassmann manifold.
        if X is not None:
            num_X = len(X)
            if num_X < 2:
                raise ValueError('UQpy: At least two matrices must be provided.')

            plist = []
            for i in range(num_X):
                [n, m] = np.shape(X[i])
                plist.append(min(n,m))

            test_p = all(elem == plist[0] for elem in plist)

            if test_p:
                self.p = plist[0]
            else:
                raise ValueError('UQpy: the elements in X are in different manifolds.')

            self.X = X

    def gradient_descent(self, **kwargs):

        """
        Compute the Karcher mean using the gradient descent method.

        This method computes the Karcher mean given a set of points on the Grassmann manifold. In this regard, the
        ``gradient_descent`` method is implemented herein also considering the acceleration scheme due to Nesterov.
        Further, this method is called by the method ``karcher_mean``.

        **Input:**

        * **data_points** (`list`)
            Points on the Grassmann manifold.

        * **distance_fun** (`callable`)
            Distance function.

        * **kwargs** (`dictionary`)
            Contains the keywords for the used in the optimizers to find the Karcher mean.

        **Output/Returns:**

        * **mean_element** (`list`)
            Karcher mean.

        """

        data_points = self.X

        # acc is a boolean varible to activate the Nesterov acceleration scheme.
        if 'acc' in kwargs.keys():
            acc = kwargs['acc']
        else:
            acc = False

        # Error tolerance
        if 'tol' in kwargs.keys():
            tol = kwargs['tol']
        else:
            tol = 1e-3

        # Maximum number of iterations.
        if 'maxiter' in kwargs.keys():
            maxiter = kwargs['maxiter']
        else:
            maxiter = 1000

        # Number of points.
        n_mat = len(data_points)

        # =========================================
        alpha = 0.5
        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(data_points[i])))

        max_rank = max(rnk)
        fmean = []
        for i in range(n_mat):
            fmean.append(self.frechet_variance(data_points[i]))

        index_0 = fmean.index(min(fmean))
        mean_element = data_points[index_0].tolist()

        avg_gamma = np.zeros([np.shape(data_points[0])[0], np.shape(data_points[0])[1]])

        itera = 0

        l = 0
        avg = []
        _gamma = []
        if acc:
            _gamma = self.log_map(points_grassmann=data_points, ref=np.asarray(mean_element))

            avg_gamma.fill(0)
            for i in range(n_mat):
                avg_gamma += _gamma[i] / n_mat
            avg.append(avg_gamma)

        # Main loop
        while itera <= maxiter:
            _gamma = self.log_map(points_grassmann=data_points, ref=np.asarray(mean_element))
            avg_gamma.fill(0)

            for i in range(n_mat):
                avg_gamma += _gamma[i] / n_mat

            test_0 = np.linalg.norm(avg_gamma, 'fro')
            if test_0 < tol and itera == 0:
                break

            # Nesterov: Accelerated Gradient Descent
            if acc:
                avg.append(avg_gamma)
                l0 = l
                l1 = 0.5 * (1 + np.sqrt(1 + 4 * l * l))
                ls = (1 - l0) / l1
                step = (1 - ls) * avg[itera + 1] + ls * avg[itera]
                l = copy.copy(l1)
            else:
                step = alpha * avg_gamma

            x = self.exp_map(points_tangent=[step], ref=np.asarray(mean_element))

            test_1 = np.linalg.norm(x[0] - mean_element, 'fro')

            if test_1 < tol:
                break

            mean_element = []
            mean_element = x[0]

            itera += 1

        # return the Karcher mean.
        self.karcher_mean = mean_element

    def stochastic_gradient_descent(self, **kwargs):

        """
        Compute the Karcher mean using the stochastic gradient descent method.

        This method computes the Karcher mean given a set of points on the Grassmann manifold. In this regard, the
        ``stochastic_gradient_descent`` method is implemented herein. Further, this method is called by the method
        ``karcher_mean``.

        **Input:**

        * **data_points** (`list`)
            Points on the Grassmann manifold.

        * **distance_fun** (`callable`)
            Distance function.

        * **kwargs** (`dictionary`)
            Contains the keywords for the used in the optimizers to find the Karcher mean.

        **Output/Returns:**

        * **mean_element** (`list`)
            Karcher mean.
        """

        data_points = self.X

        if 'tol' in kwargs.keys():
            tol = kwargs['tol']
        else:
            tol = 1e-3

        if 'maxiter' in kwargs.keys():
            maxiter = kwargs['maxiter']
        else:
            maxiter = 1000

        n_mat = len(data_points)

        rnk = []
        for i in range(n_mat):
            rnk.append(min(np.shape(data_points[i])))

        max_rank = max(rnk)

        fmean = []
        for i in range(n_mat):
            fmean.append(self.frechet_variance(data_points[i]))

        index_0 = fmean.index(min(fmean))

        mean_element = data_points[index_0].tolist()
        itera = 0
        _gamma = []
        k = 1
        while itera < maxiter:

            indices = np.arange(n_mat)
            np.random.shuffle(indices)

            melem = mean_element
            for i in range(len(indices)):
                alpha = 0.5 / k
                idx = indices[i]
                _gamma = self.log_map(points_grassmann=[data_points[idx]], ref=np.asarray(mean_element))

                step = 2 * alpha * _gamma[0]

                X = self.exp_map(points_tangent=[step], ref=np.asarray(mean_element))

                _gamma = []
                mean_element = X[0]

                k += 1

            test_1 = np.linalg.norm(mean_element - melem, 'fro')
            if test_1 < tol:
                break

            itera += 1

        self.karcher_mean = mean_element

    def frechet_variance(self, y):

        """
        Compute the Frechet variance.

        The Frechet variance corresponds to the summation of the square distances, on the manifold, to a given
        point also on the manifold. This method is employed in the minimization scheme used to find the Karcher mean.

        **Input:**

        * **point_grassmann** (`list` or `ndarray`)
            Point on the Grassmann manifold where the Frechet variance is computed.

        * **points_grassmann** (`list` or `ndarray`)
            Points on the Grassmann manifold.

        * **distance_fun** (`callable`)
            Distance function.

        **Output/Returns:**

        * **frechet_var** (`list`)
            Frechet variance.

        """

        data_points = self.X
        distance_fun = self.distance_object.distance

        nargs = len(data_points)

        if nargs < 2:
            raise ValueError('UQpy: At least two input matrices must be provided.')

        accum = 0
        for i in range(nargs):
            d = distance_fun(y, data_points[i])
            accum += d ** 2

        frechet_var = accum / nargs

        return frechet_var
