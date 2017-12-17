# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
import utilscv
import static as st

# A Python class has been created, called ImageFeature
# that will contain for each of the images in the database,
# the information needed to compute the recognition of objects.
class ImageFeature(object):
    def __init__(self, nameFile, shape, imageBinary, kp, desc):
        # File name
        self.nameFile = nameFile
        # Shape of the image
        self.shape = shape
        # Binary data of the image
        self.imageBinary = imageBinary
        # KeyPoints of the image once the feature detection algorithm is applied
        self.kp = kp
        # Descriptors of the detected features
        self.desc = desc
        # Matchings of the image of the database with the image of the webcam
        self.matchingWebcam = []
        # Matching the webcam with the current image of the database.
        self.matchingDatabase = []
    # Allows you to empty previously calculated matching for a new image
    def clearMatchingMutuals(self):
        self.matchingWebcam = []
        self.matchingDatabase = []


class ObjectRecognitionHelper:

    def __init__(self):
        self.opencv_utils = utilscv.OpenCvUtils()

    # Functions responsible for calculating, for each of the calculation methods of features,
    # the features of each one of the images of the "models" directory
    def loadModelsFromDirectory(self):
        # The method returns a dictionary. The key is the features algorithm
        # while the value is a list with objects of type ImageFeature
        # where all the data of the features of the images of
        # Database.
        dataBase = []
        # The number of features has been limited to 250, so that the algorithm goes smoothly.
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=250)
        for imageFile in os.listdir(st.MODELS_DIRECTORY_NAME):
            # The image is loaded with the OpenCV
            colorImage = cv2.imread(st.MODELS_DIRECTORY_NAME + "/" + str(imageFile))
            # We pass the image in grayscale
            currentImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY)
            # We make a resize of the image, so that the compared image is equal
            kp, desc = sift.detectAndCompute(currentImage, None)
            # The features are loaded with SIFT
            dataBase.append(ImageFeature(imageFile, currentImage.shape, colorImage, kp, desc))
        return dataBase

    # Function responsible for calculating mutual Matching, but nesting loops
    # It's a very slow solution because it does not take advantage of Numpy power
    # We do not even put a slider to use this method since it is very slow
    def findMatchingMutuals(self, selectedDataBase, desc, kp):
        for imgFeatures in selectedDataBase:
            imgFeatures.clearMatchingMutuals()
            for i in range(len(desc)):
                firstMatching = None
                canditateDataBase = None
                matchingSecond = None
                candidateWebCam = None
                for j in range(len(imgFeatures.desc)):
                    valueMatching = np.linalg.norm(desc[i] - imgFeatures.desc[j])
                    if (firstMatching is None or valueMatching < firstMatching):
                        firstMatching = valueMatching
                        canditateDataBase = j
                for k in range(len(desc)):
                    valueMatching = np.linalg.norm(imgFeatures.desc[canditateDataBase] - desc[k])
                    if (matchingSecond is None or valueMatching < matchingSecond):
                        matchingSecond = valueMatching
                        candidateWebCam = k
                if not candidateWebCam is None and i == candidateWebCam:
                    imgFeatures.matchingWebcam.append(kp[i].pt)
                    imgFeatures.matchingDatabase.append(imgFeatures.kp[canditateDataBase].pt)
        return selectedDataBase

    # Function responsible for calculating the mutual matching of a webcam image,
    # with all the images of the database. Receive as input parameter
    # the database based on the feature calculation method used
    # in the webcam entry image.
    def findMatchingMutualsOptimizated(self, selectedDataBase, desc, kp):
        # The algorithm is repeated for each image in the database.
        for img in selectedDataBase:
            img.clearMatchingMutuals()
            for i in range(len(desc)):
                # The standard of the difference of the current descriptor is calculated, with all
                # the descriptors of the image of the database. We get
                # without loops and using Numpy broadcasting, all distances
                # between the current descriptor and all descriptors of the current image
                distanceListFromWebCam = np.linalg.norm(desc[i] - img.desc, axis=-1)
                # The candidate that is less than the current descriptor is obtained
                candidateDataBase = distanceListFromWebCam.argmin()
                # It checks if the matching is mutual, that is, if it is met
                # in the other sense. That is, it is verified that the candidateDatabase
                # has the current descriptor as the best matching
                distanceListFromDataBase = np.linalg.norm(img.desc[candidateDataBase] - desc, axis=-1)
                candidateWebCam = distanceListFromDataBase.argmin()
                # If the mutual matching is fulfilled, it is stored for later treatment
                if (i == candidateWebCam):
                    img.matchingWebcam.append(kp[i].pt)
                    img.matchingDatabase.append(img.kp[candidateDataBase].pt)
            # For convenience they become Numpy ND-Array
            img.matchingWebcam = np.array(img.matchingWebcam)
            img.matchingDatabase = np.array(img.matchingDatabase)
        return selectedDataBase

    # This function calculates the best image according to the number of inliers
    # that has each image of the database with the image obtained from
    # the web camera.
    def calculateBestImageByNumInliers(self, selectedDataBase, projer, minInliers):
        if minInliers < 15:
            minInliers = 15
        bestIndex = None
        bestMask = None
        numInliers = 0
        # For each of the images
        for index, imgWithMatching in enumerate(selectedDataBase):
            # The RANSAC algorithm is computed to calculate the number of inliers
            _, mask = cv2.findHomography(imgWithMatching.matchingDatabase,
                                         imgWithMatching.matchingWebcam, cv2.RANSAC, projer)
            if not mask is None:
                # Check the number of inliers from the mask.
                # If the number of inliers is higher than the minimum number of inliers,
                # and it is a maximum (it has more inliers than the previous image),
                # then it is considered to be the image that matches the object
                # stored in the database.
                countNonZero = np.count_nonzero(mask)
                if (countNonZero >= minInliers and countNonZero > numInliers):
                    numInliers = countNonZero
                    bestIndex = index
                    bestMask = (mask >= 1).reshape(-1)
        # If an image has been obtained as the best image and, therefore,
        # must have a minimum number of inlers, then you calculate finally
        # the keypoints that are inliers from the mask obtained in findHomography
        # and it is returned as the best image.
        if not bestIndex is None:
            bestImage = selectedDataBase[bestIndex]
            inliersWebCam = bestImage.matchingWebcam[bestMask]
            inliersDataBase = bestImage.matchingDatabase[bestMask]
            return bestImage, inliersWebCam, inliersDataBase
        return None, None, None

    # This function calculates the affinity matrix A, paints a rectangle around
    # of the detected object and it paints in a new window the image of the database
    # corresponding to the recognized object.
    def calculateAffinityMatrixAndDraw(self, bestImage, inliersDataBase, inliersWebCam, imgout):
        # The affinity matrix A is calculated
        A = cv2.estimateRigidTransform(inliersDataBase, inliersWebCam, fullAffine=True)
        A = np.vstack((A, [0, 0, 1]))

        # The points of the rectangle occupied by the recognized object are calculated
        a = np.array([0, 0, 1], np.float)
        b = np.array([bestImage.shape[1], 0, 1], np.float)
        c = np.array([bestImage.shape[1], bestImage.shape[0], 1], np.float)
        d = np.array([0, bestImage.shape[0], 1], np.float)
        center = np.array([float(bestImage.shape[0])/2,
           float(bestImage.shape[1])/2, 1], np.float)

        # The points of the virtual space are multiplied, to convert them into
        # real points of the image
        a = np.dot(A, a)
        b = np.dot(A, b)
        c = np.dot(A, c)
        d = np.dot(A, d)
        center = np.dot(A, center)

        # The points are dehomogenized
        areal = (int(a[0]/a[2]), int(a[1]/b[2]))
        breal = (int(b[0]/b[2]), int(b[1]/b[2]))
        creal = (int(c[0]/c[2]), int(c[1]/c[2]))
        dreal = (int(d[0]/d[2]), int(d[1]/d[2]))
        centroreal = (int(center[0]/center[2]), int(center[1]/center[2]))

        # The polygon and the name of the image file are painted in the center of the polygon
        points = np.array([areal, breal, creal, dreal], np.int32)
        cv2.polylines(imgout, np.int32([points]),1, (255,255,255), thickness=2)
        self.opencv_utils.draw_str(imgout, centroreal, bestImage.nameFile.upper())
        # The detected object is displayed in a separate window
        cv2.imshow(st.MODEL_FRAME_NAME, bestImage.imageBinary)
