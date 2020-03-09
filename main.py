import argparse
import astropy
import numpy as np
import cv2 as cv
from astropy.io import fits
from scipy import ndimage
import matplotlib.pyplot as plt
from anytree import Node, RenderTree, LevelOrderIter
from pathlib import Path
from skimage import exposure
from astropy.visualization import simple_norm
import background_extraction as bge
from astropy.modeling import models, fitting
import warnings
import astropy.stats

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
            else: #turns out this path may never get hit due to the way opencv orders their list?
                node_i = Node(str(i), parent=cnt_nodes[H[i][3]])



    print(H)
    return root

def int_tup (in_tup):
    return (int(in_tup[0]), int(in_tup[1]))


def create_display_img(img):
    norm = simple_norm(img,'sqrt')

    display_img = np.stack((norm(img),) * 3, axis=-1)


    return display_img


def get_defocussed_stars(img, debug_imgs=False):





    print("making grid")
    y, x = np.mgrid[:img.shape[0], :img.shape[1]]

    # Fit the data using astropy.modeling
    p_init = models.Polynomial2D(degree=2)
    fit_lsq = fitting.LinearLSQFitter()
    fit_sigma = fitting.FittingWithOutlierRemoval(fit_lsq, astropy.stats.sigma_clip, niter=3,
                                                  sigma=3.0)
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        print("Fitting")
        p, mask = fit_sigma(p_init, x, y, img)
    bg = p(x, y)

    if debug_imgs:
        # Plot the data with the best-fit model
        plt.figure(figsize=(8, 2.5))
        plt.subplot(1, 3, 1)
        plt.imshow(img, origin='lower', interpolation='nearest')
        plt.title("Data")
        plt.subplot(1, 3, 2)
        plt.imshow(bg, origin='lower', interpolation='nearest')
        plt.title("Model")
        plt.subplot(1, 3, 3)
        plt.imshow(img - bg, origin='lower', interpolation='nearest')
        plt.title("Residual")
        plt.show()

    corrected = (img - bg).astype(np.single)
    filtered = cv.bilateralFilter(corrected, d=19, sigmaColor=10000, sigmaSpace=10000)

    binary_img = filtered > 3 * np.std(filtered) + np.median(filtered)
    if debug_imgs:
        plt.imshow(binary_img)
        plt.show()
    uint8_img = (binary_img * 1).astype(np.uint8)

    im2, contours, hierarchy = cv.findContours(uint8_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #now its time to get from pixels to contours
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
        display_img = create_display_img(img)

        cv.drawContours(display_img, contours, -1, (0, 1, 0), 3)

        for node in LevelOrderIter(tree):
            if node is not tree and hasattr(node,'elip'):
                cv.ellipse(display_img, node.elip, (1,0,0), 2)

        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Detected stars')
        ax.imshow(display_img)
        plt.show()

    return tree

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def line_points_from_elip_axis(elip, length):
    centre = elip[0]
    angle = elip[2]
    print(length)
    p = pol2cart(length, np.radians(angle))

    return (int_tup(centre), int_tup((p[0] + centre[0], p[1] + centre[1])))


def analyse_off_axis(img, stars_tree, debug_imgs=False):


    final_img = create_display_img(img)

    for node in LevelOrderIter(stars_tree, maxlevel=2):
        if node is not stars_tree and hasattr(node, 'elip'):
            axes = node.elip[1]
            minor, major = axes
            ecc = np.sqrt(1- ( (minor**2)/ (major**2) ))
            print(ecc)
            line_pts = line_points_from_elip_axis(node.elip, length=200*ecc)
            print("Drawing pointer: " + str(line_pts))
            cv.line(final_img, line_pts[0], line_pts[1],color=(0,1,0), thickness=2)

    plt.imshow(final_img)
    plt.show()





def analyse_on_axis(img, stars_tree, debug_imgs=False):

    # now find ellipse with its centre closest to the centre of the image
    centre_star = None
    centre_coords = (img.shape[1] / 2, img.shape[0] / 2) # we want an xy point from shape which is yx
    current_shortest_dist = float("inf")
    for node in LevelOrderIter(stars_tree, maxlevel=2):
        if node is not stars_tree and hasattr(node, 'elip') and len(node.children) > 0:
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
        display_img = create_display_img(img)

        for node in LevelOrderIter(centre_star):
            if hasattr(node, 'elip'):
                cv.ellipse(display_img, node.elip, (0, 0, 1), 2)

        cv.line(display_img, int_tup(centre_coords), int_tup(centre_star.elip[0]),
                color=(0, 1, 0), thickness=2)

        cv.line(display_img, int_tup(centre_star.elip[0]), int_tup(centre_obstruction.elip[0]),
                color=(1, 0, 0), thickness=2)
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Centre star')
        ax.imshow(display_img)
        plt.show()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-fits_path", required=True)
    parser.add_argument('-on_axis', action='store_true')
    parser.add_argument('-off_axis', action='store_true')
    parser.add_argument('-debug', action='store_true')


    args = parser.parse_args()
    print(args)

    fits_path = Path(args.fits_path)

    fits_file = fits.open(str(fits_path))
    img = fits_file[0].data
    img = img.astype('single')

    stars = get_defocussed_stars(img, True)

    if args.on_axis:
        analyse_on_axis(img.copy(), stars, debug_imgs=args.debug)

    if args.off_axis:
        analyse_off_axis(img.copy(), stars, debug_imgs=args.debug)




if __name__ == '__main__':
    main()