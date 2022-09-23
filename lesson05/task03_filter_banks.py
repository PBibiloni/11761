import math
import typing as tp

import cv2
import numpy as np
from matplotlib import pyplot as plt, colors

from utils import sample_filepath


def kernel_gaussian(sigma, filter_size=(11, 11)):
    """Returns a Gaussian kernel."""
    # Your code here:
    # ...
    kernel = np.zeros(filter_size)
    for i in range(filter_size[0]):
        for j in range(filter_size[1]):
            x = i - filter_size[0] // 2
            y = j - filter_size[1] // 2
            r = np.sqrt(x ** 2 + y ** 2)
            kernel[i, j] = np.exp(-(r ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def kernel_gabor(sigma: float, theta: float, lambd: float, gamma: float, filter_size=(11, 11)) -> np.ndarray:
    """Returns a Gabor kernel."""
    # Your code here: see `cv2.getGaborKernel()`
    # ...
    return cv2.getGaborKernel(ksize=filter_size, sigma=sigma, theta=theta, lambd=lambd, gamma=gamma, psi=0)


def kernel_derivative(dx: int, dy: int, filter_size=(11, 11)) -> np.ndarray:
    """Returns kernels for computing derivatives of an image (dx, dy are derivative order wrt x and y respectively)."""
    # Your code here: use `cv2.getDerivKernels(..., dx= , dy= , ...)` to compute second-order derivatives
    #                 ...but be careful: it returns two 1D array (why? use `cv2.filter2D` to reconstruct the kernel).
    # ...
    dx, dy = cv2.getDerivKernels(dx=dx, dy=dy, ksize=5, kx=filter_size[0], ky=filter_size[1])

    kernel = np.zeros(filter_size)
    kernel[filter_size[0] // 2, filter_size[1] // 2] = 1
    kernel = cv2.filter2D(kernel, cv2.CV_64F, kernel=dx)
    kernel = cv2.filter2D(kernel, cv2.CV_64F, kernel=dy.T)
    return kernel


def kernel_laplacian_of_gaussian(sigma, filter_size=(11, 11)):
    """Returns a Laplacian of Gaussian kernel."""
    # Your code here: you can use `cv2.Sobel(..., dx= , dy= , ...)` to compute second-order derivatives
    #                 remember that laplacian(f) = dx^2/d^2 f + dy^2/d^2 f
    gaussian = kernel_gaussian(sigma, filter_size)
    return cv2.Sobel(gaussian, cv2.CV_64F, dx=2, dy=0) + cv2.Sobel(gaussian, cv2.CV_64F, dx=0, dy=2)


def kernel_schmidt(sigma, tau, filter_size=(11, 11)):
    """Returns a Schmidt (isotropic wave) kernel."""
    # Your code here: remember that F(sigma, tau) = exp(-r^2 / (2 * sigma^2)) * cos(pi * tau * r / sigma).
    # ...
    kernel = np.zeros(filter_size)
    for i in range(filter_size[0]):
        for j in range(filter_size[1]):
            x = i - filter_size[0] // 2
            y = j - filter_size[1] // 2
            # Your code here: remember that F(sigma, tau) = exp(-r^2 / (2 * sigma^2)) * cos(pi * tau * r / sigma).
            # ...
            r = np.sqrt(x ** 2 + y ** 2)
            kernel[i, j] = np.exp(-(r ** 2) / (2 * sigma ** 2)) * np.cos(np.pi * tau * r / sigma)
    return kernel


if __name__ == "__main__":
    img = cv2.imread(sample_filepath('Retinal_DRIVE21_original.tif'), cv2.IMREAD_COLOR)
    img = img[150:200, 100:150, :]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernels = {}
    kernels.update({
        f'dx^{i}': kernel_derivative(dx=i, dy=0)
        for i in range(1, 4)
    })
    kernels.update({
        f'Gs[s={sigma}]': kernel_gaussian(sigma=sigma)
        for sigma in range(1, 4)
    })
    kernels.update({
        f'Gb[th={theta:0.1f}]': kernel_gabor(sigma=3, theta=theta, lambd=1.5, gamma=1)
        for theta in [np.pi, np.pi/2, np.pi/4, np.pi*4/5, 0]
    })
    kernels.update({
        f'LG[s={sigma}]': kernel_laplacian_of_gaussian(sigma=sigma)
        for sigma in range(1, 4)
    })
    kernels.update({
        f'S[s={sigma},t={tau}]': kernel_schmidt(sigma=sigma, tau=tau)
        for sigma in [1, 2, 3]
        for tau in [1, 2, 3]
    })
    kernels = [(k, v) for k, v in kernels.items() if v is not None]

    # Visualize
    for idx in range(math.ceil(len(kernels)/4)):
        _, axs = plt.subplots(2, 5)
        axs[0, 0].imshow(img_gray, cmap='gray')
        axs[1, 0].set_axis_off()
        for idx, (name, kernel) in enumerate(kernels[idx*4:idx*4+4]):
            img_filtered = cv2.filter2D(img_gray, cv2.CV_64F, kernel)
            axs[0, idx+1].imshow(kernel, cmap='gray')
            axs[1, idx+1].set_title(name)
            axs[1, idx+1].imshow(img_filtered, cmap='gray')
        plt.show()

