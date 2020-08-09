import matplotlib.pyplot as plt
import numpy as np

from skopt.space import Integer, Real
from skopt.utils import use_named_args
from skopt.plots import plot_objective
from skopt.plots import plot_convergence
from skopt import gp_minimize
from matplotlib import axes




dim_sims = Integer(name='simulations', low=10, high=40)
dim_dcoeff = Real(name='dcoeff', low=0.01, high=0.03)
dim_c = Integer(name='c', low=3, high=5)
dimensions = [dim_sims, dim_dcoeff, dim_c]

@use_named_args(dimensions)
def obj_func(simulations, dcoeff, c):
    res = np.random.uniform(low=0.0, high=1.0, size=None)
    print("res = ", res)
    return res # we are trying to minimize this

hp_game_count = 28
results = gp_minimize(obj_func, dimensions=dimensions, n_calls=10, acq_func='EI', x0=None, y0=None, noise=1e-8)
print("min f(x) after n calls is : ", results.fun)
wins = hp_game_count-int(results.fun)*hp_game_count
print("Best params: {} wins: {}".format(results.x,  wins))
plot_convergence(results, yscale='log').figure.savefig("convergence.png")
plot_objective(results).flatten()[0].figure.savefig("objective.png")



#axes.Axes.imshow(plot_convergence(results, yscale='log'))
#axes.savefig("filename.png")

plot_objective(results)

x = np.linspace(0, 20, 100)  # Create a list of evenly-spaced numbers over the range
plt.plot(x, np.sin(x))       # Plot the sine of each x point
plt.show()
a =2
b = 3

print("END a+b=", a+b)