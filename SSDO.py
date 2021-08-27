from numpy import array, mean, sin, cos
from numpy.random import uniform
from copy import deepcopy
from optimizer import Optimizer


class BaseSSDO(Optimizer):
    """
    The original version of: Social Ski-Driver Optimization (SSDO)
        (Parameters optimization of support vector machines for imbalanced data using social ski driver algorithm)
    Noted:
        https://doi.org/10.1007/s00521-019-04159-z
        https://www.mathworks.com/matlabcentral/fileexchange/71210-social-ski-driver-ssd-optimization-algorithm-2019
    """

    def __init__(self, obj_func=None, lb=None, ub=None, verbose=True, epoch=150, pop_size=10, **kwargs):
        super().__init__(problem = {
                "obj_func": obj_func,
                "lb": lb,
                "ub": ub,
                "epoch": epoch,
                "verbose": verbose,
                "pop_size": pop_size,})
        self.epoch = epoch
        self.pop_size = pop_size #Number of individuals / Number of solutions
        self.obj_func = obj_func
        self.lb = lb
        self.ub = ub
        self.verbose = verbose

    def train(self):
        pop = [self.create_solution() for _ in range(self.pop_size)]
        g_best = self.get_global_best_solution(pop)

        list_velocity = uniform(self.lb, self.ub, (self.pop_size, self.problem_size))
        list_local_best = array([item[self.ID_POS] for item in pop])

        for epoch in range(self.epoch):
            c = 2 - epoch * (2.0 / self.epoch)  # a decreases linearly from 2 to 0

            # Update Position based on velocity
            for i in range(0, self.pop_size):
                pos_new = pop[i][self.ID_POS] + list_velocity[i]
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                if fit_new < pop[i][self.ID_FIT]:
                    list_local_best[i] = deepcopy(pos_new)
                pop[i] = [pos_new, fit_new]

            ## Calculate the mean of the best three solutions in each dimension. Eq 9
            pop = sorted(pop, key=lambda item: item[self.ID_FIT])
            pop_best_3 = deepcopy(pop[:3])
            pos_list_3 = array([item[self.ID_POS] for item in pop_best_3])
            pos_mean = mean(pos_list_3)

            # Updating velocity vectors
            for i in range(0, self.pop_size):
                r1 = uniform()  # r1, r2 is a random number in [0,1]
                r2 = uniform()
                if r2 <= 0.5:     ## Use Sine function to move
                    vel_new = c * sin(r1) * (list_local_best[i] - pop[i][self.ID_POS]) + sin(r1) * (pos_mean - pop[i][self.ID_POS])
                else:                   ## Use Cosine function to move
                    vel_new = c * cos(r1) * (list_local_best[i] - pop[i][self.ID_POS]) + cos(r1) * (pos_mean - pop[i][self.ID_POS])
                list_velocity[i] = deepcopy(vel_new)

            # Update the global best
            g_best = self.update_global_best_solution(pop)
            self.loss_train.append(g_best[self.ID_FIT])
            if self.verbose:
                print(">Epoch: {}, Best fit: {}".format(epoch + 1, g_best[self.ID_FIT]))
        self.solution = g_best
        return g_best[self.ID_POS], g_best[self.ID_FIT], self.loss_train
