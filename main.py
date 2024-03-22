import math
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from colormap import colormap

np.random.seed(10)


TIME_STEP = .01
STEPS_PER_FRAME = 2
GD_COLOR = "#0e4c62"
HMC_COLOR = "#46014f"


def main():
	x = np.linspace(-0.9, 1.3, 201)
	y = np.linspace(-0.55, 0.6, 101)
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

	os.makedirs("results/", exist_ok=True)
	for tag, show in [("gradient", {"GD"}), ("hamiltonian", {"HMC"}), ("both", {"GD", "HMC"})]:
		os.makedirs(f"results/{tag}-frames/", exist_ok=True)
		for filename in os.listdir(f"results/{tag}-frames/"):
			os.remove(f"results/{tag}-frames/{filename}")

		plt.figure(figsize=((x[-1] - x[0])*3 + .4, (y[-1] - y[0])*3 + .4))
		plt.imshow(
			np.exp(-Z).T, extent=(
				x[0] - (x[1] - x[0])/2, x[-1] + (x[1] - x[0])/2,
				y[0] - (y[1] - y[0])/2, y[-1] + (y[1] - y[0])/2,
			),
			cmap=colormap, vmin=0, vmax=1, origin="lower", zorder=10)
		plt.contour(
			x, y, np.sqrt(Z).T, levels=np.arange(0, np.max(Z), .3),
			colors="k", linewidths=.5, zorder=20)
		line_GD, = plt.plot([], [], GD_COLOR, zorder=30)
		dot_GD = None
		line_HMC, = plt.plot([], [], HMC_COLOR, zorder=32)
		dots_HMC = None
		plt.xlim(x[0], x[-1])
		plt.ylim(y[0], y[-1])
		plt.axis("off")
		plt.tight_layout()
		for i in range(0, len(points_HMC), STEPS_PER_FRAME):
			if "GD" in show and i < len(points_GD):
				line_GD.set_xdata(points_GD[:i + 1, 0])
				line_GD.set_ydata(points_GD[:i + 1, 1])
				if dot_GD is not None:
					dot_GD.remove()
				dot_GD = plt.scatter(
					points_GD[i, 0], points_GD[i, 1],
					c=GD_COLOR, marker="v", zorder=31)
			if "HMC" in show:
				line_HMC.set_xdata(points_HMC[:i + 1, 0])
				line_HMC.set_ydata(points_HMC[:i + 1, 1])
				indices = np.concatenate([np.arange(0, i, 50), [i]])
				if dots_HMC is not None:
					dots_HMC.remove()
				dots_HMC = plt.scatter(
					points_HMC[indices, 0], points_HMC[indices, 1],
					c=HMC_COLOR, marker="o", zorder=33)
			plt.savefig(f"results/{tag}-frames/{i//STEPS_PER_FRAME:03d}.png")
			plt.pause(.1)
		plt.savefig(f"results/{tag}-static.png")
		plt.savefig(f"results/{tag}-static.svg")
		plt.close()


def gradient_descent(grad_x_func, grad_y_func, start):
	steps_per_step = 10
	state = np.array(start)
	history = [start]
	for i in range(math.ceil(3.0/TIME_STEP*steps_per_step)):
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
