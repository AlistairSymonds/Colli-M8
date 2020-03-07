import argparse
import astropy
import numpy as np
import cv2 as cv
from astropy.io import fits
from scipy import ndimage
import matplotlib.pyplot as plt
from anytree import Node, RenderTree, LevelOrderIter


#from https://stackoverflow.com/questions/33964913/equivalent-of-polyfit-for-a-2d-polynomial-in-python
def polyfit2d(x, y, z, kx=3, ky=3, order=None):
    '''
    Two dimensional polynomial fitting by least squares.
    Fits the functional form f(x,y) = z.

    Notes
    -----
    Resultant fit can be plotted with:
    np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

    Parameters
    ----------
    x, y: array-like, 1d
        x and y coordinates.
    z: np.ndarray, 2d
        Surface to fit.
    kx, ky: int, default is 3
        Polynomial order in x and y, respectively.
    order: int or None, default is None
        If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
        If int, coefficients up to a maximum of kx+ky <= order are considered.

    Returns
    -------
    Return paramters from np.linalg.lstsq.

    soln: np.ndarray
        Array of polynomial coefficients.
    residuals: np.ndarray
    rank: int
    s: np.ndarray

    '''

    # grid coords
    x, y = np.meshgrid(x, y)
    # coefficient array, up to x^kx, y^ky
    coeffs = np.ones((kx+1, ky+1))

    # solve array
    a = np.zeros((coeffs.size, x.size))

    # for each coefficient produce array x^i, y^j
    for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
        # do not include powers greater than order
        if order is not None and i + j > order:
            arr = np.zeros_like(x)
        else:
            arr = coeffs[i, j] * x**i * y**j
        a[index] = arr.flatten()

    # do leastsq fitting and return leastsq result
    return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

def histogram_img(img):
    sigma = np.std(img)
    mean = np.mean(img)
    print("Sigma: " + str(sigma))

    neg_sigmas = (np.mean(img) - np.min(img) )/ sigma
    pos_sigmas = (np.max(img) - np.mean(img) )/ sigma

    print(neg_sigmas)
    print(np.min(img))
    print(pos_sigmas)
    print(np.max(img))

    low_samples = (img > (mean - sigma))
    high_samples = (img < (mean + sigma))
    mean_ish_img = low_samples & high_samples
    #plt.imshow(mean_ish_img)
    #plt.show()

def cvhier2tree(hierarchy, contours):
    root = Node(name='root', parent=None)
    H = hierarchy[0]

    cnt_nodes = [None] * len(contours)
    #hierarchy fields:
    # 0 = next contour idx
    # 1 = previous contour idx
    # 2 = first child idx
    # 3 = parent idx
    for i in range(len(H)):
        if H[i][3] == -1:
            node_i = Node(str(i), parent=root, cnt=contours[i])
            cnt_nodes[i] = node_i
        else:
            if cnt_nodes[H[i][3]] != None:
                node_i = Node(str(i), parent=cnt_nodes[H[i][3]], cnt=contours[i])
                cnt_nodes[i] = node_i
            else: #turns out this path may not get hit due to the way opencv orders their list?
                node_i = Node(str(i), parent=cnt_nodes[H[i][3]])



    print(H)
    #print(cnt_nodes)
    #print(RenderTree(root))
    return root


def get_defocussed_stars(img, debug_imgs=False):
    binary_img = img > np.std(img) + np.median(img)
    if debug_imgs:
        plt.imshow(binary_img)
        plt.show()
    uint8_img = (binary_img * 1).astype(np.uint8)

    im2, contours, hierarchy = cv.findContours(uint8_img, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)
    if debug_imgs:
        plt.imshow(binary_img)
        plt.show()

    tree = cvhier2tree(hierarchy,contours)


    for node in LevelOrderIter(tree, maxlevel=2):
        if node is not tree:
            if node.cnt.shape[0] < 5:
                node.parent = None


    #first we will remove all top level nodes that do not fit an ellipse well in terms of area
    for node in LevelOrderIter(tree, maxlevel=2):
        if node is not tree:
            elip = cv.fitEllipse(node.cnt)
            A = np.pi * elip[1][0]/2 * elip[1][1]/2 # Area = pi * semi major axis * semi minor axis
            cA = cv.contourArea(node.cnt)
            print("Ellipse area = "+str(A) + " Contour Area = " + str(cA))
            if  0.9*A < cA and cA < 1.1*cA:
                node.elip = elip
            else:
                print("removing node centred at " + str(elip[0]))



    if debug_imgs:
        contour_img = np.zeros((uint8_img.shape[0], uint8_img.shape[1], 3))
        contour_img = cv.normalize(img, cv.NORM_MINMAX, 0, 255)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Star blob detection')
        ax.imshow(contour_img)
        plt.show()
        for node in LevelOrderIter(tree):
            if node is not tree and hasattr(node,'elip'):
                cv.ellipse(contour_img, node.elip, (255), 2)

        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Star blob detection')
        ax.imshow(contour_img)
        plt.show()


def analyse_off_axis(img, debug_imgs=False):
    img = img.astype('single')
    mat_blurred = cv.bilateralFilter(img, d=19, sigmaColor=50000, sigmaSpace=50000)

    histogram_img(mat_blurred)
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    kx = 1
    ky = 1
    bg = polyfit2d(x, y, mat_blurred, kx=kx, ky=ky)
    coeffs = bg[0].reshape((kx + 1, ky + 1))
    fitted_surf = np.polynomial.polynomial.polygrid2d(x, y, coeffs)

    mat_corrected = mat_blurred - fitted_surf

    stars_tree = get_defocussed_stars(mat_corrected, debug_imgs=debug_imgs)

    if debug_imgs:
        fig, ax = plt.subplots(2, 2)
        fig.suptitle('Background extraction')
        ax[0, 0].imshow(img)
        ax[0, 1].imshow(mat_blurred)
        ax[1, 0].imshow(fitted_surf)
        ax[1, 1].imshow(mat_corrected)
        plt.show()



def analyse_on_axis(img, debug_imgs=False):
    get_defocussed_stars(img, debug_imgs=debug_imgs)



def main():
    on_axis = cv.imread('C:/Users/alist/OneDrive/Development/Colli-M8/on-axis-test-img.png', cv.IMREAD_GRAYSCALE)
    analyse_on_axis(on_axis, debug_imgs=True)

    img = fits.open("C:/NINA_images/2020-02-28_collimation/LIGHT/2020-02-29_01-36-59_Lum_-5.80_20.00s_0002.fits")
    mat = img[0].data
    analyse_off_axis(mat, debug_imgs=True)




if __name__ == '__main__':
    main()