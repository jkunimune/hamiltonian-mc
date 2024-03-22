import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from colormap import colormap

np.random.seed(10)


TIME_STEP = .01


def main():
	x = np.linspace(-0.9, 1.3, 201)
	y = np.linspace(-0.55, 0.6, 201)
	X, Y = np.meshgrid(x, y, indexing="ij")
	angle = .25
	X_rot = X*np.cos(angle) - (Y + .1)*np.sin(angle)
	Y_rot = X*np.sin(angle) + (Y + .1)*np.cos(angle)
	Y_valley = X_rot**2/2 + X_rot**3/6
	Z = 20*(Y_rot - Y_valley)**2 + 2*X_rot**2
	cost_func = interpolate.RectBivariateSpline(x, y, Z)
	grad_x_func = cost_func.partial_derivative(1, 0)
	grad_y_func = cost_func.partial_derivative(0, 1)

	start = (0.9, 0.4)

	points_GD = gradient_descent(grad_x_func, grad_y_func, start)
	points_HMC = hamiltonian_monte_carlo(cost_func, grad_x_func, grad_y_func, start)

	plt.imshow(
		np.exp(-Z).T, extent=(
			x[0] - (x[1] - x[0])/2, x[-1] + (x[1] - x[0])/2,
			y[0] - (y[1] - y[0])/2, y[-1] + (y[1] - y[0])/2,
		),
		cmap=colormap, vmin=0, vmax=1, origin="lower", zorder=10)
	plt.contour(
		x, y, np.sqrt(Z).T, levels=np.arange(0, np.max(Z), .3),
		colors="k", linewidths=.5, zorder=20)
	plt.plot(points_GD[:, 0], points_GD[:, 1], "#0c2766", zorder=30)
	plt.scatter(points_GD[::2, 0], points_GD[::2, 1], c="#0c2766", zorder=31)
	plt.plot(points_HMC[:, 0], points_HMC[:, 1], "#0c2766", zorder=32)
	plt.scatter(points_HMC[::50, 0], points_HMC[::50, 1], c="#0c2766", zorder=33)
	plt.xlim(x[0], x[-1])
	plt.ylim(y[0], y[-1])
	plt.xticks([])
	plt.yticks([])
	plt.show()


def gradient_descent(grad_x_func, grad_y_func, start):
	steps_per_step = 10
	state = np.array(start)
	history = [start]
	for i in range(math.ceil(4.5/TIME_STEP*steps_per_step)):
		gradient = np.array([grad_x_func(*state)[0, 0], grad_y_func(*state)[0, 0]])
		velocity = -gradient/max(.5, np.hypot(*gradient))
		state = state + velocity*TIME_STEP/steps_per_step
		if (i + 1)%steps_per_step == 0:
			history.append(state)
	return np.array(history)


def hamiltonian_monte_carlo(potential_func, force_x_func, force_y_func, start):
	steps_per_step = 10
	steps_per_collision = 50
	state = np.array(start)
	history = [state]
	gradient = np.array([force_x_func(*state)[0, 0], force_y_func(*state)[0, 0]])
	velocity = -gradient/np.hypot(*gradient)
	while len(history) < math.ceil(4.5/TIME_STEP):
		old_state = state
		old_energy = 1/2*np.sum(velocity**2) + potential_func(*state)[0, 0]
		for j in range(math.ceil(steps_per_collision*steps_per_step)):
			gradient = np.array([force_x_func(*state)[0, 0], force_y_func(*state)[0, 0]])
			state = state + velocity*TIME_STEP/steps_per_step/2
			velocity = velocity - gradient*TIME_STEP/steps_per_step
			state = state + velocity*TIME_STEP/steps_per_step/2
			if (j + 1)%steps_per_step == 0:
				history.append(state)
		new_energy = 1/2*np.sum(velocity**2) + potential_func(*state)[0, 0]
		a = min(1, np.exp(old_energy - new_energy))
		if np.random.random() >= a:
			state = old_state
		velocity = np.random.normal(0, 1, 2)
	return np.array(history)


if __name__ == "__main__":
	main()
