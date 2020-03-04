import argparse
import astropy
import numpy as np
import cv2 as cv
from astropy.io import fits
from scipy import ndimage
import matplotlib.pyplot as plt


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


    if debug_imgs:
        fig, ax = plt.subplots(2, 2)
        fig.suptitle('Background extraction')
        ax[0, 0].imshow(img)
        ax[0, 1].imshow(mat_blurred)
        ax[1, 0].imshow(fitted_surf)
        ax[1, 1].imshow(mat_corrected)
        plt.show()

    binary_img = mat_corrected >  np.std(mat_corrected) + np.median(mat_corrected)
    plt.imshow(binary_img)
    plt.show()
    countour_img = (binary_img*1).astype(np.uint8)

    im2, contours, hierarchy = cv.findContours(countour_img, cv.RETR_TREE, cv.CHAIN_APPROX_TC89_L1)

    approx_contours = []
    for cnt in contours:
        approx_contours.append(cv.approxPolyDP(cnt, closed=True, epsilon=0.2))

    contours_one_child_policy = []
    H = hierarchy[0]
    for i in range(len(contours)):
        #is conoutour top level and have at least one child?
        parent = H[i][3]
        first_child = H[i][2]
        if H[i][3] == -1 and H[i][2] >= 0:
            #there are no other contours at this level
            #if H[first_child][0] == -1:
            contours_one_child_policy.append(contours[i])




    if debug_imgs:
        contour_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3))
        for cnt in contours:
            if cnt.shape[0] >= 5:
                cv.drawContours(contour_img, [cnt], 0, (255,0,0), 1)
                elip = cv.fitEllipse(cnt)
                cv.ellipse(contour_img,elip,(0,255,0),2)

        one_child_img = np.zeros((binary_img.shape[0], binary_img.shape[1], 3))
        for cnt in contours_one_child_policy:
            if cnt.shape[0] >= 5:
                cv.drawContours(one_child_img, [cnt], 0, (255,0,0), 1)
                elip = cv.fitEllipse(cnt)
                cv.ellipse(one_child_img, elip, (0, 255, 0), 2)



        fig, ax = plt.subplots(1, 2)
        fig.suptitle('Star blob detection')
        ax[0].imshow(contour_img)
        ax[1].imshow(one_child_img)
        plt.show()

def analyse_on_axis(img, debug_imgs=False):
    binary_img = img > np.std(img) + np.median(img)
    plt.imshow(binary_img)
    plt.show()
    countour_img = (binary_img * 1).astype(np.uint8)

    im2, contours, hierarchy = cv.findContours(countour_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    convex_contours = []
    for cnt in contours:
        if cv.isContourConvex(cnt):
            convex_contours.append(cnt)

    detected_stars = np.zeros(img.shape, dtype=np.uint8)
    print(len(convex_contours))
    for cnt in convex_contours:
        cv.drawContours(detected_stars, [cnt], 0, 255, 3)

    if debug_imgs:
        fig, ax = plt.subplots(1, 3)
        fig.suptitle('Star blob detection')
        ax[0].imshow(img)
        ax[1].imshow(detected_stars)
        plt.show()



def main():
    on_axis = cv.imread('C:/Users/alist/OneDrive/Development/Colli-M8/on-axis-test-img.png', cv.IMREAD_GRAYSCALE)
    #analyse_on_axis(on_axis)

    img = fits.open("C:/NINA_images/2020-02-28_collimation/LIGHT/2020-02-29_01-36-59_Lum_-5.80_20.00s_0002.fits")
    mat = img[0].data
    analyse_off_axis(mat, debug_imgs=True)




if __name__ == '__main__':
    main()