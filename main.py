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
    return root

def int_tup (in_tup):
    return (int(in_tup[0]), int(in_tup[1]))

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

    #we only care about nodes we can fit an ellipse to so, so remove the ones we can't
    for node in LevelOrderIter(tree):
        if node is not tree:
            if node.cnt.shape[0] < 5:
                node.parent = None
            else:
                node.elip = cv.fitEllipse(node.cnt)


    #first we will remove all top level nodes that do not fit an ellipse well in terms of area
    for node in LevelOrderIter(tree, maxlevel=2):
        if node is not tree:

            A = np.pi * node.elip[1][0]/2 * node.elip[1][1]/2 # Area = pi * semi major axis * semi minor axis
            cA = cv.contourArea(node.cnt)
            print("Ellipse area = "+str(A) + " Contour Area = " + str(cA))
            if  0.9*A > cA or cA > 1.1*cA:
                node.parent = None
                print("removing node centred at " + str(node.elip[0]))

    if debug_imgs:
        contour_img = np.zeros((uint8_img.shape[0], uint8_img.shape[1], 1))
        contour_img = cv.normalize(img, contour_img, 0, 255, cv.NORM_MINMAX)
        display_img = np.ascontiguousarray(np.moveaxis(np.array([contour_img, contour_img, contour_img]), 0, 2))


        for node in LevelOrderIter(tree):
            if node is not tree and hasattr(node,'elip'):
                cv.ellipse(display_img, node.elip, (255,0,0), 2)

        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Detected stars')
        ax.imshow(display_img)
        plt.show()

    return tree




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
    tree = get_defocussed_stars(img, debug_imgs=debug_imgs)

    # now find ellipse with its centre closest to the centre of the image
    centre_star = None
    centre_coords = (img.shape[0] / 2, img.shape[1] / 2)
    current_shortest_dist = float("inf")
    for node in LevelOrderIter(tree, maxlevel=2):
        if node is not tree and hasattr(node, 'elip'):
            dist = cv.norm(centre_coords, node.elip[0], cv.NORM_L2)
            print("Smallest dist so far = " + str(
                current_shortest_dist) + " current ellipse centre dist = " + str(dist))
            if dist < current_shortest_dist:
                centre_star = node
                current_shortest_dist = dist

    # if not refractor :P (find centre hole from central obstruction))
    centre_obstruction = None
    current_biggest_area = float("-inf")
    for node in LevelOrderIter(centre_star, maxlevel=2):
        if node is not centre_star and hasattr(node, 'elip'):
            A = np.pi * node.elip[1][0] / 2 * node.elip[1][1] / 2
            print("Biggest are so far = " + str(
                current_shortest_dist) + " current ellipse area = " + str(A))
            if A > current_biggest_area:
                centre_obstruction = node
                current_shortest_dist = A

    if debug_imgs:
        contour_img = np.zeros((img.shape[0], img.shape[1], 1))
        contour_img = cv.normalize(img, contour_img, 0, 255, cv.NORM_MINMAX)
        display_img = np.ascontiguousarray(
            np.moveaxis(np.array([contour_img, contour_img, contour_img]), 0, 2))

        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Centre Star')
        ax.imshow(contour_img)
        plt.show()
        for node in LevelOrderIter(centre_star):
            if hasattr(node, 'elip'):
                cv.ellipse(display_img, node.elip, (0, 0, 255), 2)

        cv.line(display_img, int_tup(centre_coords), int_tup(centre_star.elip[0]),
                color=(0, 255, 0), thickness=2)

        cv.line(display_img, int_tup(centre_star.elip[0]), int_tup(centre_obstruction.elip[0]),
                color=(255, 0, 0), thickness=2)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Centre star')
        ax.imshow(display_img)
        plt.show()


def main():
    on_axis_fits = fits.open('data/eigen/_L_SNAPSHOT_2020-03-05_21-19-14_0000_8.00s_-15.00_0.00.fits')
    on_axis = on_axis_fits[0].data

    dsi = fits.open("data/example_balanced.fit")
    img = dsi[0].data

    analyse_on_axis(img, debug_imgs=True)


    analyse_off_axis(img, debug_imgs=True)




if __name__ == '__main__':
    main()