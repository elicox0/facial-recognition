# image_segmentation.py
"""
Elijah Cox
11/2020
"""

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from scipy import sparse
from imageio import imread
import matplotlib.pyplot as plt 


def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    return np.diag(np.sum(A,axis=1)) - A


def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    evals,_ = [np.real(i) for i in la.eig(A)]
    check_tol = lambda x: 0 if x < tol else x
    evals = [check_tol(x) for x in evals]
    components = len(evals) - np.count_nonzero(evals)
    alg_connectivity = np.partition(evals,1)[1]

    return components, alg_connectivity


def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


class ImageSegmenter:
    """Class for storing and segmenting images."""

    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        self.image = imread(filename) / 255
        if len(self.image.shape) == 3:
            brightness = self.image.mean(axis=2)
            self.gray = False
        else:
            brightness = self.image
            self.gray = True
        self.brightness = np.ravel(brightness)

    def show_original(self):
        """Display the original image."""
        plt.axis('off')
        if self.gray == True:
            plt.imshow(self.image, cmap='gray')
        else:
            plt.imshow(self.image)
        plt.show()

    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        # Initialize A and D
        m,n = self.image.shape[:2]
        mn = m*n
        A = sparse.lil_matrix((mn,mn))
        D = np.zeros(mn)
        
        # For each vertex, find the set of vertices within r, and corresponding weights. (Equation 5.3)
        for i in range(mn):
            J, distances = get_neighbors(i,r,m,n)
            weights = [np.exp( -abs(self.brightness[i]-self.brightness[j])/sigma_B2 
                               -distances[k]/sigma_X2 ) for k,j in enumerate(J)]
            A[i,J] = weights                               
            D[i] = sum(weights)
        return A.tocsc(), D

    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        L = sparse.csgraph.laplacian(A)
        Ds = sparse.diags(1/np.sqrt(D)).tocsc()
        DLD = Ds @ L.dot(Ds)
        eigs = spla.eigsh(DLD, which='SM',k=2)
        m = np.argmax(eigs[0])
        v = eigs[1][:, m].reshape(self.image.shape[:2])
        return v > 0

    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A,D = self.adjacency(r=r,sigma_B2=sigma_B,sigma_X2=sigma_X)
        mask = self.cut(A,D)

        # Original image
        plt.subplot(131)
        plt.axis('off')
        if self.gray == True:
            plt.imshow(self.image, cmap='gray')
        else:
            plt.imshow(self.image)

        # Positive segment
        plt.subplot(132)
        plt.axis('off')
        if self.gray == True:
            plt.imshow(self.image * mask, cmap='gray')
        else:
            plt.imshow(self.image * np.dstack((mask,mask,mask)))

        # Negative segment
        plt.subplot(133)
        plt.axis('off')
        if self.gray == True:
            plt.imshow(self.image * ~mask, cmap='gray')
        else:
            plt.imshow(self.image * np.dstack((~mask,~mask,~mask)))

        plt.show()
