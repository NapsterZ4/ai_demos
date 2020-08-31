import numpy as np
import cv2
import streamlit as st


def normalize_histograms(im):  # normalizes the histogram of images
    im1 = im.copy()
    for i in range(3):
        imi = im[:, :, i]
        # print(imi.shape)
        minval = np.min(imi)
        maxval = np.max(imi)
        # print(minval,maxval)
        imrange = maxval - minval
        im1[:, :, i] = (255 / (imrange + 0.0001) * (
                imi - minval))  # imi-minval will turn the color range between 0-imrange, and the scaleing will stretch the range between 0-255
    return im1


######################################################################
# This following function reads the images from file,
# auto crops the image to its relevant content, then normalizes
# the histograms of the cropped images
######################################################################

def read_and_process_image(filename):
    path = "/mnt/napster_disk/ai_projects/demos/breast_cancer/analizer.png"
    im_array = np.array(filename)
    cv2.imwrite(path, cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR))
    im = cv2.imread(path)

    # The following steps re needed for auto cropping the black paddings in the images

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  # convert 2 grayscale
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # turn it into a binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    if len(contours) != 0:
        # find the biggest area
        cnt = max(contours, key=cv2.contourArea)

        # find the bounding rect
        x, y, w, h = cv2.boundingRect(cnt)

        crop = im[y:y + h, x:x + w]  # crop image
        # crop1=cv2.resize(crop,(im_size,im_size)) # resize to im_size X im_size size
        crop = normalize_histograms(crop)
        return crop
    else:
        return (normalize_histograms(im))


##################################################################################
#### The following functions are for extracting features from the images #########
##################################################################################

# histogram statistics (mean, standard deviations, energy, entropy, log-kurtosis)

def histogram_statistics(hist):
    # hist= cv2.calcHist([gr],[0],None,[256],[0,256])
    hist = hist / np.sum(hist)  # probabilities
    hist = hist.reshape(-1)
    hist[hist == 0] = 10 ** -20  # replace zeros with a small number
    mn = np.sum([i * hist[i] for i in range(len(hist))])  # mean
    std_dev = np.sqrt(np.sum([((i - mn) ** 2) * hist[i] for i in range(len(hist))]))  # standard deviation
    energy = np.sum([hist[i] ** 2 for i in range(len(hist))])  # energy
    entropy = np.sum([hist[i] * np.log(hist[i]) for i in range(len(hist))])  # entropy
    kurtosis = np.log(np.sum([(std_dev ** -4) * ((i - mn) ** -4) * hist[i] for i in range(len(hist))]))  # kurtosis
    return [mn, std_dev, energy, entropy, kurtosis]


#################################################################
# create thresholding based features, the idea is to hand engineer some features based on adaptive thresholding.
# After looking at the images it appeared  that adaptive thresholding may
# leave different artifacts in the processed images, we can extract several features from these artifacts
##################################################################

def thresholding_based_features(im, imsize, quartiles):
    im = cv2.resize(im, (imsize, imsize))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    w = 11  # window
    t = 5  # threshold
    counts = []
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, w,
                               t)  # adaptive gaussian threshold the image
    th = cv2.bitwise_not(
        th)  # invert the image (the black pixels will turn white and the white pixels will turn black)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)  # find cntours in the image
    # print(len(contours))

    q = np.zeros(len(quartiles))  # quartiles of contours will be stored here

    for cnt in contours:
        area = cv2.contourArea(cnt)  # calculate the area of the contours
        if area < 40000:  # Exclude contours that are too big, generally these are the image outlines
            counts.append(area)
    if len(counts) > 1:
        q = np.quantile(np.array(counts), quartiles)  # contour quartiles

    return (q, len(counts), np.sum(th) / (255 * th.shape[0] * th.shape[
        1]))  # return the contour quartiles, number of contours, proportion of white pixels in the thresholded images
    # counts.append(np.sum(th)/(normalizing_factor*(th.shape[0]*th.shape[1])))


@st.cache(allow_output_mutation=True)
def various_features(image_ml):
    B = []
    G = []
    R = []

    # mini 16 bin histograms
    hist_B = []
    hist_G = []
    hist_R = []

    # statistics fom full 256 bin shitogram
    hist_feat_B = []
    hist_feat_G = []
    hist_feat_R = []
    hist_feat_GS = []

    # thresholding based features
    mean_pixels = []  # proportion of white pixels
    contour_quartiles = []  # contour area quartiles
    no_of_contours = []  # total number of contours

    quartiles = np.arange(0.1, 1, 0.1)  # contour area quartiles
    bins = 16  # mini histogram bins

    im = read_and_process_image(image_ml)
    # im_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    # im_yuv[:,:,0] = cv2.equalizeHist(im_yuv[:,:,0])

    # convert the YUV image back to RGB format
    # im = cv2.cvtColor(im_yuv, cv2.COLOR_YUV2BGR)

    # median color
    B.append(np.median(im[:, :, 0]))
    G.append(np.median(im[:, :, 1]))
    R.append(np.median(im[:, :, 2]))

    # histograms
    hist_B.append(cv2.calcHist([im], [0], None, [bins], [0, 256]) / (im.size / 3))
    hist_G.append(cv2.calcHist([im], [1], None, [bins], [0, 256]) / (im.size / 3))
    hist_R.append(cv2.calcHist([im], [2], None, [bins], [0, 256]) / (im.size / 3))

    # more histogram features

    hist_feat_B.append(histogram_statistics(cv2.calcHist([im], [0], None, [256], [0, 256])))
    hist_feat_G.append(histogram_statistics(cv2.calcHist([im], [1], None, [256], [0, 256])))
    hist_feat_R.append(histogram_statistics(cv2.calcHist([im], [2], None, [256], [0, 256])))

    gr = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gr = cv2.equalizeHist(gr)
    hist_feat_GS.append(histogram_statistics(cv2.calcHist([gr], [0], None, [256], [0, 256])))

    # threshold featues
    q, nc, m = thresholding_based_features(im, 256, quartiles)
    mean_pixels.append(m)
    contour_quartiles.append(q)
    no_of_contours.append(nc)

    # create feature vectors
    width_of_features = 3 * bins + len(quartiles) + 2 + 20  # 20 features are histogram statistics

    X = np.zeros((1, width_of_features))  # this is where all features will be stored

    X[0, 0:bins] = hist_B[0].reshape(-1)
    X[0, bins:2 * bins] = hist_G[0].reshape(-1)
    X[0, 2 * bins:3 * bins] = hist_R[0].reshape(-1)
    X[0, 3 * bins:3 * bins + len(quartiles)] = contour_quartiles[0].reshape(-1)
    X[0, 3 * bins + len(quartiles)] = mean_pixels[0]
    X[0, 3 * bins + len(quartiles) + 1] = no_of_contours[0]
    start = 3 * bins + len(quartiles) + 2
    X[0, start:start + 5] = hist_feat_B[0]
    X[0, start + 5:start + 10] = hist_feat_G[0]
    X[0, start + 10:start + 15] = hist_feat_R[0]
    X[0, start + 15:start + 20] = hist_feat_B[0]

    return X