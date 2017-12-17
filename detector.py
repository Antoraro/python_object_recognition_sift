# -*- coding: utf-8 -*-

import sys
import cv2
import time

import videoinput
import utilscv
import objrecogn
import static as st


class Detector:

    def __init__(self):
        cv2.namedWindow(st.MAIN_FRAME_NAME)
        cv2.namedWindow(st.MODEL_FRAME_NAME)

        # Opening video source:
        if len(sys.argv) > 1:
            strsource = sys.argv[1]
        else:
            strsource = '0:rows=300:cols=400'  # Simple apertura de la cámara cero, sin escalado
        self.videoinput = videoinput.VideoInput(strsource)
        self.paused = False
        self.orec = objrecogn.ObjectRecognitionHelper()
        self.opencv_utils = utilscv.OpenCvUtils()

        # We load the database of the models
        self.directoryDataBase = self.orec.loadModelsFromDirectory()
        # Creation of the detector of features (only at the beginning):
        self.detector = cv2.xfeatures2d.SIFT_create(nfeatures=250)

    def draw_frame(self, imgout, kp, dim, t1):
        # We write informative text about the image
        self.opencv_utils.draw_str(imgout, (20, 20),
                         "Method {0}, {1} features found, desc. dim. = {2} ".
                         format(st.METHOD_NAME, len(kp), dim))
        self.opencv_utils.draw_str(imgout, (20, 40), "Time (ms): {0}".format(str(t1)))
        # Show results and check keys:
        cv2.imshow('Detector', imgout)

    def close_all_views(self):
        # Close window (s) and video source (s):
        self.videoinput.close()
        cv2.destroyAllWindows()

    def do_detection(self):
        while True:
            # Reading input frame, and interface parameters:
            if not self.paused:
                frame = self.videoinput.read()
            if frame is None:
                print('End of video input')
                break

            # We pass input image to grays:
            imgin = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # The output image is calculated
            imgout = frame.copy()
            # We detect features, and measure time:
            t1 = time.time()
            kp, desc = self.detector.detectAndCompute(imgin, None)
            if len(self.directoryDataBase) > 0:
                # We perform mutual matching
                imgsMatchingMutuals = self.orec.findMatchingMutualsOptimizated(self.directoryDataBase, desc, kp)
                minInliers = 20
                projer = 5.0
                # The best image is calculated according to the number of inliers.
                # The best image is the one with the most number of inliers, but always
                # exceeding the minimum indicated in the trackbar 'minInliers'
                bestImage, inliersWebCam, inliersDataBase = self.orec.calculateBestImageByNumInliers(self.directoryDataBase,
                                                                                                projer, minInliers)
                if not bestImage is None:
                    # If we find a good image, the affinity matrix is calculated and the recognized object is painted on the screen.
                    self.orec.calculateAffinityMatrixAndDraw(bestImage, inliersDataBase, inliersWebCam, imgout)

            t1 = 1000 * (time.time() - t1)  # Time in milliseconds
            # Obtain dimension of descriptors for each feature:
            if desc is not None:
                if len(desc) > 0:
                    dim = len(desc[0])
                else:
                    dim = -1

            self.draw_frame(imgout=imgout, kp=kp, dim=dim, t1=t1)

            ch = cv2.waitKey(5) & 0xFF
            if ch == 27:  # Escape exits
                break
            elif ch == ord(' '):  # Space bar pauses
                self.paused = not self.paused
            elif ch == ord('.'):  # Dot checks current frame
                self.paused = True
                frame = self.videoinput.read()

        self.close_all_views()


if __name__ == '__main__':
    detector = Detector()
    detector.do_detection()
