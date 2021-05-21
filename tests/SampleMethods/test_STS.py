from UQpy.SampleMethods import RectangularStrata, RectangularSTS, VoronoiStrata, VoronoiSTS, DelaunayStrata, DelaunaySTS
from UQpy.Distributions import Exponential
import numpy as np
import os

marginals = [Exponential(loc=1., scale=1.), Exponential(loc=1., scale=1.)]
strata = RectangularStrata(nstrata=[3, 3])

x_sts = RectangularSTS(dist_object=marginals, strata_object=strata, nsamples_per_stratum=1, random_state=1)

x_sts1 = RectangularSTS(dist_object=marginals, strata_object=strata, nsamples_per_stratum=1,  sts_criterion="centered",
                        random_state=1)

dir_path = os.path.dirname(os.path.realpath(__file__))
filepath=os.path.join(dir_path,'strata.txt')
strata1 = RectangularStrata(input_file=filepath)
x_sts2 = RectangularSTS(dist_object=marginals, strata_object=strata1, nsamples_per_stratum=1, random_state=1)

strata_vor = VoronoiStrata(nseeds=8, dimension=2, random_state=3)
sts_vor = VoronoiSTS(dist_object=marginals, strata_object=strata_vor)
sts_vor.run(nsamples_per_stratum=1)

seeds = np.array([[0, 0], [0.4, 0.8], [1, 0], [1, 1]])
strata_del = DelaunayStrata(seeds=seeds, random_state=2)
sts_del = DelaunaySTS(dist_object=marginals, strata_object=strata_del)
sts_del.run(nsamples_per_stratum=2)


# Unit tests
def test_1():
    """
    Test output attributes of x_sts object
    """
    tmp1 = (np.round(x_sts.samples, 3) == np.array([[1.15, 1.275], [1.406, 1.106], [2.257, 1.031], [1.064, 1.595],
                                                    [1.627, 1.719], [2.642, 1.825], [1.071, 4.203], [1.419, 3.209],
                                                    [2.639, 2.917]])).all()
    tmp2 = (np.round(x_sts.samplesU01, 3) == np.array([[0.139, 0.24], [0.333, 0.101], [0.716, 0.031], [0.062, 0.449],
                                                       [0.466, 0.513], [0.806, 0.562], [0.068, 0.959], [0.342, 0.89],
                                                       [0.806, 0.853]])).all()
    assert tmp1 and tmp2


def test_2():
    """
    Test output attributes of x_sts2 object
    """
    tmp1 = (np.round(x_sts2.samples, 3) == np.array([[1.234, 1.275], [1., 1.569], [1.076, 2.196], [1.899, 1.19],
                                                    [1.914, 2.467], [2.93, 2.849]])).all()
    tmp2 = (np.round(x_sts2.samplesU01, 3) == np.array([[0.209, 0.24], [0., 0.434], [0.073, 0.697], [0.593, 0.173],
                                                       [0.599, 0.769], [0.855, 0.843]])).all()
    assert tmp1 and tmp2


def test_3():
    """
        Test error checks.
    """
    tmp = False
    try:
        RectangularSTS(dist_object=marginals, strata_object=strata, nsamples_per_stratum=1, sts_criterion="center")
    except NotImplementedError:
        tmp = True

    assert tmp


def test_4():
    """
        Test error checks.
    """
    tmp = False
    try:
        RectangularSTS(dist_object=marginals, strata_object=None, nsamples_per_stratum=1, sts_criterion="center")
    except NotImplementedError:
        tmp = True

    assert tmp


def test_5():
    """
        Test error checks.
    """
    tmp = False
    try:
        RectangularSTS(dist_object=marginals, strata_object=strata, nsamples_per_stratum=1, sts_criterion="centered",
                       nsamples=10)
    except ValueError:
        tmp = True

    assert tmp


def test_6():
    """
        Test error checks.
    """
    tmp1 = (np.round(sts_vor.samples, 3) == np.array([[1.519, 2.507], [1.2, 2.531], [4.642, 3.16], [1.194, 1.305],
                                                      [1.223, 1.461], [1.003, 1.689], [1.585, 1.077],
                                                      [2.126, 1.665]])).all()
    tmp2 = (np.round(sts_vor.weights, 3) == np.array([0.161, 0.158, 0.097, 0.128, 0.032, 0.054, 0.249, 0.122])).all()
    assert tmp1 and tmp2


def test_7():
    """
        Test error checks.
    """
    tmp1 = (np.round(sts_del.samples, 3) == np.array([[1.2, 2.441], [1.089, 1.778], [2.219, 1.154], [1.935, 1.077],
                                                      [1.689, 5.188], [3.11, 5.136], [1.92, 2.719],
                                                      [2.037, 2.707]])).all()
    tmp2 = (np.round(sts_del.weights, 3) == np.array([0.1, 0.1, 0.2, 0.2, 0.05, 0.05, 0.15, 0.15])).all()
    assert tmp1 and tmp2
