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


def create_display_img(img):
    norm = simple_norm(img,'sqrt')

    display_img = np.stack((norm(img),) * 3, axis=-1)


    return display_img


def get_defocussed_stars(img, debug_imgs=False):

    binary_img = img > np.std(img) + np.median(img)
    if debug_imgs:
        plt.imshow(binary_img)
        plt.show()
    uint8_img = (binary_img * 1).astype(np.uint8)

    im2, contours, hierarchy = cv.findContours(uint8_img, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)


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

def line_points_from_elip_axis(elip):
    centre = elip[0]
    axis = [1]
    angle = elip[2]

    p = pol2cart(4, angle)

    return (int_tup(centre), int_tup(p))


def analyse_off_axis(img, debug_imgs=False):
    mat_blurred = cv.bilateralFilter(img, d=19, sigmaColor=50000, sigmaSpace=50000)

    bounds = bge.gen_sample_pts(img,1,0)
    print(bounds)
    bg_samples = (bounds[0] < img) & (img < bounds[1])
    num_samples = np.sum(bg_samples)
    print("Got " + str(num_samples) + " out of " + str(len(bg_samples)) + "(" + str(num_samples/len(bg_samples)) +"%)")
    plt.imshow(bg_samples)
    plt.show()
    x_sample_coords = []
    y_sample_coords = []
    samples = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if bounds[0] < img[i,j] and img[i,j] < bounds[1]:
                x_sample_coords.append(i)
                y_sample_coords.append(j)
                samples.append(img[i,j])

    kx = 2
    ky = 2
    bg = bge.polyfit2d(x_sample_coords, y_sample_coords, samples, kx=kx, ky=ky)
    coeffs = bg[0].reshape((kx + 1, ky + 1))
    x = np.arange(img.shape[0])
    y = np.arange(img.shape[1])
    fitted_surf = np.polynomial.polynomial.polygrid2d(x, y, coeffs)

    mat_corrected = mat_blurred - fitted_surf

    stars_tree = get_defocussed_stars(img, debug_imgs=debug_imgs)
    final_img = create_display_img(img)
    for node in LevelOrderIter(stars_tree, maxlevel=2):
        if node is not stars_tree and hasattr(node, 'elip') and len(node.children) > 0:
            line_pts = line_points_from_elip_axis(node.elip)
            cv.line(final_img, line_pts[0], line_pts[1],color=(0,1,0))



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
        if node is not tree and hasattr(node, 'elip') and len(node.children) > 0:
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

    if args.on_axis:
        analyse_on_axis(img.copy(), debug_imgs=args.debug)

    if args.off_axis:
        analyse_off_axis(img.copy(), debug_imgs=args.debug)




if __name__ == '__main__':
    main()