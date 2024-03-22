import math

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


np.random.seed(2)


TIME_STEP = .01


def main():
	x = np.linspace(-1, 1, 201)
	y = np.linspace(-1, 1, 201)
	X, Y = np.meshgrid(x, y, indexing="ij")
	angle = .9
	X_rot = X*np.cos(angle) - (Y + .1)*np.sin(angle)
	Y_rot = X*np.sin(angle) + (Y + .1)*np.cos(angle)
	Y_valley = -.4 + X_rot**2/2 + X_rot**3/6
	Z = 20*(Y_rot - Y_valley)**2 + 2*X_rot**2
	cost_func = interpolate.RectBivariateSpline(x, y, Z)
	grad_x_func = cost_func.partial_derivative(1, 0)
	grad_y_func = cost_func.partial_derivative(0, 1)

	start = (.5, -.5)

	points_GD = gradient_descent(grad_x_func, grad_y_func, start)
	points_HMC = hamiltonian_monte_carlo(grad_x_func, grad_y_func, start)

	plt.imshow(np.exp(-Z).T, extent=(-1.005, 1.005, -1.005, 1.005), vmin=0, vmax=1, origin="lower", zorder=10)
	plt.contour(x, y, np.sqrt(Z).T, levels=np.arange(0, np.max(Z), .3), colors="k", linewidths=1, zorder=20)
	plt.plot(points_GD[:, 0], points_GD[:, 1], "-", zorder=30)
	plt.plot(points_GD[::2, 0], points_GD[::2, 1], "o", zorder=31)
	plt.plot(points_HMC[:, 0], points_HMC[:, 1], "-", zorder=32)
	plt.plot(points_HMC[::50, 0], points_HMC[::50, 1], "o", zorder=33)
	plt.axis([-1, 1, -1, 1])
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


def hamiltonian_monte_carlo(grad_x_func, grad_y_func, start):
	steps_per_step = 10
	steps_per_collision = 50
	state = np.array(start)
	history = [state]
	gradient = np.array([grad_x_func(*state)[0, 0], grad_y_func(*state)[0, 0]])
	velocity = -gradient/np.hypot(*gradient)
	for i in range(math.ceil(4.5/TIME_STEP/steps_per_collision)):
		for j in range(math.ceil(steps_per_collision*steps_per_step)):
			gradient = np.array([grad_x_func(*state)[0, 0], grad_y_func(*state)[0, 0]])
			state = state + velocity*TIME_STEP/steps_per_step/2
			velocity = velocity - gradient*TIME_STEP/steps_per_step
			state = state + velocity*TIME_STEP/steps_per_step/2
			if (j + 1)%steps_per_step == 0:
				history.append(state)
		velocity = np.random.normal(0, 1, 2)
	return np.array(history)


if __name__ == "__main__":
	main()
