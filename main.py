import math
import os

import matplotlib.pyplot as plt
from imageio.v2 import imread, mimsave
from numpy import newaxis, full, shape, uint8, linspace, meshgrid, cos, sin, arange, concatenate, array, hypot, \
	exp, random
from scipy import interpolate

from colormap import colormap

random.seed(10)


TIME_STEP = .01
STEPS_PER_FRAME = 2
GD_COLOR = "#0e4c62"
HMC_COLOR = "#46014f"


def main():
	x = linspace(-1.00, 1.25, 201)
	y = linspace(-0.50, 0.65, 101)
	X, Y = meshgrid(x, y, indexing="ij")
	angle = .25
	X_rot = X*cos(angle) - (Y + .1)*sin(angle)
	Y_rot = X*sin(angle) + (Y + .1)*cos(angle)
	Y_valley = X_rot**2/2 + X_rot**3/6
	Z = 20*(Y_rot - Y_valley)**2 + 2*X_rot**2
	cost_func = interpolate.RectBivariateSpline(x, y, Z)
	grad_x_func = cost_func.partial_derivative(1, 0)
	grad_y_func = cost_func.partial_derivative(0, 1)

	start = (0.9, 0.4)

	points_GD = gradient_descent(grad_x_func, grad_y_func, start)
	points_HMC = hamiltonian_monte_carlo(cost_func, grad_x_func, grad_y_func, start)

	os.makedirs("results/", exist_ok=True)
	for tag, plots in [("gradient", [{"GD"}]), ("hamiltonian", [{"HMC"}]), ("overlaid", [{"GD", "HMC"}]), ("stacked", [{"GD"}, {"HMC"}])]:
		os.makedirs(f"results/{tag}-frames/", exist_ok=True)
		for filename in os.listdir(f"results/{tag}-frames/"):
			os.remove(f"results/{tag}-frames/{filename}")

		fig, axeses = plt.subplots(
			len(plots), 1, squeeze=False,
			figsize=((x[-1] - x[0])*3 + .4, (y[-1] - y[0])*3*len(plots) + .4))
		line_GD, dot_GD, line_HMC, dots_HMC = None, None, None, None
		for axes, show in zip(axeses[:, 0], plots):
			axes.imshow(
				exp(-Z).T, extent=(
					x[0] - (x[1] - x[0])/2, x[-1] + (x[1] - x[0])/2,
					y[0] - (y[1] - y[0])/2, y[-1] + (y[1] - y[0])/2,
				),
				cmap=colormap, vmin=0, vmax=1, origin="lower", zorder=10)
			axes.contour(
				x, y, exp(-Z).T, levels=linspace(0, 1, 16)[0::2],
				colors="k", linewidths=.5, zorder=20)
			if "GD" in show:
				line_GD, = axes.plot([], [], GD_COLOR, zorder=30)
			if "HMC" in show:
				line_HMC, = axes.plot([], [], HMC_COLOR, zorder=32)
			axes.set_xlim(x[0], x[-1])
			axes.set_ylim(y[0], y[-1])
			axes.axis("off")
		plt.tight_layout()
		num_frames = 0
		for i in range(0, len(points_HMC), STEPS_PER_FRAME):
			if not any("HMC" in show for show in plots) and i >= len(points_GD):
				break
			for axes, show in zip(axeses[:, 0], plots):
				if "GD" in show and i < len(points_GD):
					line_GD.set_xdata(points_GD[:i + 1, 0])
					line_GD.set_ydata(points_GD[:i + 1, 1])
					if dot_GD is not None:
						dot_GD.remove()
					dot_GD = axes.scatter(
						points_GD[i, 0], points_GD[i, 1],
						c=GD_COLOR, marker="v", zorder=31)
				if "HMC" in show:
					line_HMC.set_xdata(points_HMC[:i + 1, 0])
					line_HMC.set_ydata(points_HMC[:i + 1, 1])
					indices = concatenate([arange(0, i, 50), [i]])
					if dots_HMC is not None:
						dots_HMC.remove()
					dots_HMC = axes.scatter(
						points_HMC[indices, 0], points_HMC[indices, 1],
						c=HMC_COLOR, marker="o", zorder=33)
			plt.savefig(f"results/{tag}-frames/{i//STEPS_PER_FRAME:03d}.png")
			plt.pause(.01)
			num_frames += 1

		make_gif(f"results/{tag}-frames/", f"results/{tag}-animated", num_frames, 12)
		plt.savefig(f"results/{tag}-static.png")
		plt.savefig(f"results/{tag}-static.svg")
		plt.close()


def gradient_descent(grad_x_func, grad_y_func, start):
	steps_per_step = 10
	state = array(start)
	history = [start]
	for i in range(math.ceil(1.7/TIME_STEP*steps_per_step)):
		gradient = array([grad_x_func(*state)[0, 0], grad_y_func(*state)[0, 0]])
		velocity = -gradient/max(.3, hypot(*gradient))
		state = state + velocity*TIME_STEP/steps_per_step
		if (i + 1)%steps_per_step == 0:
			history.append(state)
	return array(history)


def hamiltonian_monte_carlo(potential_func, force_x_func, force_y_func, start):
	steps_per_step = 10
	steps_per_collision = 50
	state = array(start)
	history = [state]
	gradient = array([force_x_func(*state)[0, 0], force_y_func(*state)[0, 0]])
	velocity = -gradient/hypot(*gradient)
	while len(history) < math.ceil(4.5/TIME_STEP):
		old_state = state
		old_energy = 1/2*sum(velocity**2) + potential_func(*state)[0, 0]
		for j in range(math.ceil(steps_per_collision*steps_per_step)):
			gradient = array([force_x_func(*state)[0, 0], force_y_func(*state)[0, 0]])
			state = state + velocity*TIME_STEP/steps_per_step/2
			velocity = velocity - gradient*TIME_STEP/steps_per_step
			state = state + velocity*TIME_STEP/steps_per_step/2
			if (j + 1)%steps_per_step == 0:
				history.append(state)
		new_energy = 1/2*sum(velocity**2) + potential_func(*state)[0, 0]
		a = min(1, exp(old_energy - new_energy))
		if random.random() >= a:
			state = old_state
		velocity = random.normal(0, 1, 2)
	return array(history)


def make_gif(frame_filename: str, gif_filename: str, num_frames: int, frame_rate: float):
	print("compiling GIF...")
	# load each frame and put them in a list
	frames = []
	for i in range(num_frames):
		frame = imread(f"{frame_filename}{i:03d}.png")
		rgb = frame[:, :, :3]
		alpha = frame[:, :, 3, newaxis]/255.
		frame = (rgb*alpha + 255*(1 - alpha)).astype(uint8) # remove transparency with a white background
		frames.append(frame)
	# also put a flash of white at the end if the image is transient
	if "wave" in frame_filename:
		for i in range(int(frame_rate/10)):
			frames.append(full(shape(frames[0]), 255, dtype=uint8))
	# save it all as a GIF
	mimsave(f"{gif_filename}.gif", frames, fps=frame_rate)
	print(f"saved '{gif_filename}.gif'!")


if __name__ == "__main__":
	main()
