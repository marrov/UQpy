from UQpy.sample_methods.adaptive_kriging_functions.LearningFunction import LearningFunction
import scipy.stats as stats


class ExpectedFeasibility(LearningFunction):
    """
            Expected Feasibility Function (EFF) for reliability analysis, see [6]_ for a detailed explanation.


            **Inputs:**

            * **surr** (`class` object):
                A kriging surrogate model, this object must have a ``predict`` method as defined in `krig_object` parameter.

            * **pop** (`ndarray`):
                An array of samples defining the learning set at which points the EFF is evaluated

            * **n_add** (`int`):
                Number of samples to be added per iteration.

                Default: 1.

            * **parameters** (`dictionary`)
                Dictionary containing all necessary parameters and the stopping criterion for the learning function. Here
                these include `a`, `epsilon`, and `eff_stop`.

            * **samples** (`ndarray`):
                The initial samples at which to evaluate the model.

            * **qoi** (`list`):
                A list, which contaains the model evaluations.

            * **dist_object** ((list of) ``Distribution`` object(s)):
                List of ``Distribution`` objects corresponding to each random variable.


            **Output/Returns:**

            * **new_samples** (`ndarray`):
                Samples selected for model evaluation.

            * **indicator** (`boolean`):
                Indicator for stopping criteria.

                `indicator = True` specifies that the stopping criterion has been met and the AKMCS.run method stops.

            * **eff_lf** (`ndarray`)
                EFF learning function evaluated at the new sample points.

            """
    def __init__(self, surrogate, pop, n_add, eff_a, eff_epsilon, eff_stop):
        self.surrogate = surrogate
        self.pop = pop
        self.n_add = n_add
        self.eff_a = eff_a
        self.eff_epsilon = eff_epsilon
        self.eff_stop = eff_stop

    def evaluate_function(self):

        g, sig = self.surrogate(self.pop, True)

        # Remove the inconsistency in the shape of 'g' and 'sig' array
        g = g.reshape([self.pop.shape[0], 1])
        sig = sig.reshape([self.pop.shape[0], 1])
        # reliability threshold: a_ = 0
        # EGRA method: epsilon = 2*sigma(x)
        a_, ep = self.eff_a, self.eff_epsilon * sig
        t1 = (a_ - g) / sig
        t2 = (a_ - ep - g) / sig
        t3 = (a_ + ep - g) / sig
        eff = (g - a_) * (2 * stats.norm.cdf(t1) - stats.norm.cdf(t2) - stats.norm.cdf(t3))
        eff += -sig * (2 * stats.norm.pdf(t1) - stats.norm.pdf(t2) - stats.norm.pdf(t3))
        eff += ep * (stats.norm.cdf(t3) - stats.norm.cdf(t2))
        rows = eff[:, 0].argsort()[-self.n_add:]

        stopping_criteria_indicator = False
        if max(eff[:, 0]) <= self.eff_stop:
            stopping_criteria_indicator = True

        new_samples = self.pop[rows, :]
        learning_function_values = eff[rows, :]
        return new_samples, learning_function_values, stopping_criteria_indicator
