import matplotlib.pyplot as plt
import numpy as np

def main():
	x = np.linspace(-1, 1, 201)
	y = np.linspace(-1, 1, 201)
	X, Y = np.meshgrid(x, y, indexing="ij")
	angle = .9
	X_rot = X*np.cos(angle) - (Y + .1)*np.sin(angle)
	Y_rot = X*np.sin(angle) + (Y + .1)*np.cos(angle)
	Y_valley = -.4 + X_rot**2/2 + X_rot**3/6
	Z = 20*(Y_rot - Y_valley)**2 + 2*X_rot**2

	plt.imshow(np.exp(-Z).T, extent=(-1.005, 1.005, -1.005, 1.005), vmin=0, vmax=1, origin="lower")
	plt.contour(x, y, np.sqrt(Z).T, levels=np.arange(0, np.max(Z), .3), colors="k", linewidths=1)
	plt.axis([-1, 1, -1, 1])
	plt.show()


if __name__ == "__main__":
	main()
