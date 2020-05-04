import numpy as np
import cv2 as cv
import logging
from skimage.filters import threshold_multiotsu
from sklearn.cluster import DBSCAN
from operator import attrgetter
from collections import Counter

import hilti.constants as consts
import hilti.config as cfg
from hilti.blob import Blob
from hilti.segment import Segment

from hilti.utilities import timeit
from hilti.utilities import resize_tm
from hilti.utilities import draw_circle
from hilti.utilities import scale_bounding_box


log = logging.getLogger(__name__)

"""@package hilti.Image
Package that contains Image class.

"""


class Image:
    """Image Class

    Handles IO of CV-pipeline
    """
    def __init__(self):
        """The constructor of Image class
        The init holds variables for the settings of cv pipeline. You can
        change the settings when adjusting the variables.
        """
        self.rgb_img = None
        self.gray_img = None

        self.orig_width = 0
        self.orig_height = 0
        self.mean_pixels = 0
        self.std_pixels = 0
        self.seg_median_blur = 0
        self.seg_opening_kern_size = 0
        self.seg_opening_iters = 0

        # Lazily loaded templates for template matching.
        self._templates = None
        self._templates_blurred = None

        # blob detector segments
        self._blob_detector_seg = None

        # segments
        self.total_nb_segments = 12
        self.segments = []
        self.rgb_seg = np.zeros(1, dtype=np.uint8)
        self.gray_seg = np.zeros(1, dtype=np.uint8)
        self.filter_th = 0.1  # filter threshold for filter_segments method

    @property
    def templates(self):
        """Return a cached list of pre-processed templates."""
        if self._templates is None:
            self._templates = self._load_and_preprocess_templates()
        return self._templates

    @property
    def blurred_templates(self):
        """Return a cached list of blurred templates."""
        if self._templates_blurred is None:
            self._templates_blurred = self._blur_templates()
        return self._templates_blurred

    @property
    def blob_detector_seg(self):
        if self._blob_detector_seg is None:
            params = cv.SimpleBlobDetector_Params()
            params.minDistBetweenBlobs = 5
            # params.blobColor = consts.MAX_PIXEL_VAL
            params.filterByColor = True
            params.blobColor = 255
            params.filterByArea = True
            params.minArea = 2.01
            # params.maxArea = 40
            params.filterByInertia = False
            params.filterByCircularity = True
            params.minCircularity = .6
            params.filterByConvexity = False
            params.minConvexity = .75
            self._blob_detector_seg = cv.SimpleBlobDetector_create(params)
        return self._blob_detector_seg

    @property
    def dHeight(self):
        sum_of_heights = sum(t.shape[0] for t in self.blurred_templates)
        return sum_of_heights // len(self.blurred_templates)

    @property
    def dWidth(self):
        sum_of_widths = sum(t.shape[1] for t in self.blurred_templates)
        return sum_of_widths // len(self.blurred_templates)

    def set_img(self, img_path, resize_divisor=1):
        """Set new image for cv-pipeline

        :param img_path: The path to the next image that will be processed
        :param resize_divisor: If image was resized set it to the corresponding
        value it was resized with
        :param verbose: Converts the input image to BGR for debugging purposes
        :return: No return
        """
        self.gray_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        self.init_vars(resize_divisor=resize_divisor)

    def init_vars(self, resize_divisor):
        """Initializer for cv pipeline variables

        :param resize_divisor: If image was resized set it to the corresponding
        value it was resized with
        :return: No return
        """
        self.orig_height = self.gray_img.shape[0]
        self.orig_width = self.gray_img.shape[1]
        self.mean_pixels, self.std_pixels = cv.meanStdDev(self.gray_img)

        # process segment variables
        self.seg_median_blur = 9
        self.seg_opening_kern_size = 7
        self.seg_opening_iters = 3
        self.nb_otsu_classes = 5

    def reset_vars(self):
        """Reset variables
        This method resets the variables needed to process a new image.
        :return: No return
        """
        self.rgb_img = None
        self.gray_img = None
        self.segments = []

    def _load_and_preprocess_templates(self, scaler=0.9):
        temps = [cv.imread(p, cv.IMREAD_GRAYSCALE) for p in cfg.template_path]
        temps = [cv.resize(t, (int(scaler * t.shape[1]),
                               int(scaler * t.shape[0]))) for t in temps]
        return temps

    def _blur_templates(self, blur=11, pyrs=3, verbose=False):
        temps = [cv.medianBlur(t, blur) for t in self.templates]
        for _ in range(pyrs):
            temps = [cv.pyrDown(t) for t in temps]
        return temps

    def get_img_height_width(self, div=1):
        """Get height and width of image

        :param div: Divider to get multiples of height, width
        :return: No return
        """
        # CAUTION: OpenCV returns height, width
        return self.gray_img.shape[0] // div, self.gray_img.shape[1] // div

    def show_img(self, name, src, divisor):
        """Displays an image for debugging

        :param name: Window Name
        :param src: cv.Mat (numpy array) containing the source data
        :param divisor: Divisor for the window size shown on screen.
        :return: No return
        """
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        height, width = self.get_img_height_width(div=divisor)
        cv.resizeWindow(name, width, height)
        cv.imshow(name, src)

    def assert_nmbr_segments(self):
        """Assertion statement for number of segments
        Logs an error if the number of segments detected is not equal to 12.
        :return: No return
        """
        try:
            assert (len(self.segments) == 12)
        except AssertionError:
            # ToDo write this AssertionError to log file for frontend?
            log.error("WARNING: The number of detected segments != 12"
                      " --> %i segments were detected.\n"
                      "Sorting will not work for this setup."
                      % len(self.segments))

    @timeit()
    def set_dia_segs_multi_scale_tm(self, verbose=False):
        """Multi-scale template matching
        This method sets 12 segment positions into self.segments using multi-
        scale template matching algorithm.

        :param verbose: Activate debug windows to show what's happening here
        :return: No return
        """

        if verbose:
            for i, template in enumerate(self.blurred_templates):
                self.show_img("blurred template" + str(i), template, divisor=5)

        blurred_img = cv.medianBlur(self.gray_img, 41)
        if verbose:
            self.show_img("blurred img", blurred_img, divisor=6)

        blurred_img = cv.morphologyEx(blurred_img, cv.MORPH_CLOSE,
                                      np.ones((3, 3), dtype=np.uint8),
                                      iterations=10)
        if verbose:
            self.show_img("Opening", blurred_img, divisor=6)

        # pyr down
        pyrs = 3
        for i in range(pyrs):
            blurred_img = cv.pyrDown(blurred_img)

        # multiplier for scaling
        mult = 2 ** pyrs
        for _ in range(self.total_nb_segments + 1):
            found = None
            for img_scale in np.linspace(0.8, 1.0, 6)[::-1]:
                resized = resize_tm(blurred_img,
                                    int(blurred_img.shape[1] * img_scale))
                ratio = blurred_img.shape[1] / float(resized.shape[1])
                if resized.shape[0] < self.dHeight:
                    break
                if resized.shape[1] < self.dWidth:
                    break
                for template in self.blurred_templates:
                    result = cv.matchTemplate(resized, template,
                                              cv.TM_CCOEFF_NORMED)
                    (_, maxVal, _, maxLoc) = cv.minMaxLoc(result)
                    if verbose:
                        # draw a bounding box around the detected region
                        clone = np.dstack([resized, resized, resized])
                        cv.rectangle(clone,
                                     (maxLoc[0], maxLoc[1]),
                                     (maxLoc[0] + self.dWidth,
                                      maxLoc[1] + self.dHeight),
                                     (0, 0, 255), 2)
                        cv.imshow("Visualize", clone)

                    # if we have found a new maximum correlation value, then
                    # update the bookkeeping variable
                    if found is None or maxVal > found[0]:
                        found = (maxVal, maxLoc, ratio)

            # unpack the bookkeeping varaible and compute the (x, y)
            # coordinates of the bounding box based on the resized ratio
            (acc, maxLoc, r) = found
            log.debug(" template matching max accuracy = %.1f%%", acc * 100)
            (startX, startY) = (int(maxLoc[0] * r * mult),
                                int(maxLoc[1] * r * mult))
            (endX, endY) = (int((maxLoc[0] + self.dWidth) * r * mult),
                            int((maxLoc[1] + self.dHeight) * r * mult))

            x, y, w, h = scale_bounding_box(startX, startY,
                                            endX - startX, endY - startY,
                                            scale=0.9,
                                            up_width=False, up_height=True)
            # x, y, w, h = scale_bounding_box(x, y, w, h,
            #                                 scale=0.97,
            #                                 up_width=True, up_height=False)
            self.segments.append(Segment(x, y, w, h))
            self.segments[-1].set_template_matching_accuracy(acc)

            # remove the detected segment from image
            x_big, y_big, w_big, h_big = scale_bounding_box(x, y, w, h,
                                                            scale=1.1)

            blurred_img[y_big // mult:(y_big + h_big) // mult,
                        x_big // mult:(x_big + w_big) // mult] = \
                self.mean_pixels - self.std_pixels

            if verbose:
                self.show_img("current blurred", blurred_img, divisor=4)
                cv.rectangle(self.rgb_img, (x, y),
                             (x + w, y + h),
                             (255, 255, 0), 5)
                self.show_img("multi temp match", self.rgb_img, divisor=4)
                # cv.waitKey()

    @timeit()
    def filter_segments(self, verbose=False):
        """Compares a possible 13th segment to the 12th for more robustness and
        filters based on their accuracy why one is more likely to be correct.
        Removes the segment that is less likely to be correct from the list of
        segments found during multi-scale template matching

        :param verbose: Prints debug messages
        """
        # sort by template matching accuracy (lowest first)
        self.segments.sort(key=lambda seg: seg.accuracy)
        if verbose:
            for seg in self.segments:
                log.debug(seg)

        if abs(self.segments[0].accuracy - self.segments[1].accuracy) \
                < self.filter_th:
            # test y distance to all other segments
            dy_0 = np.inf
            dy_1 = np.inf

            for seg in self.segments[2:]:
                y_0 = abs(self.segments[0].y - seg.y)
                y_1 = abs(self.segments[1].y - seg.y)

                if y_0 < dy_0:
                    dy_0 = y_0

                if y_1 < dy_1:
                    dy_1 = y_1

            if dy_0 < dy_1:
                self.segments.pop(1)
            else:
                self.segments.pop(0)
        else:
            self.segments.pop(0)

    @timeit()
    def sort_segments(self, verbose=False):
        """Segment sorter
        This method sorts the segment by a specific order starting with segment
        on the top left and ending bottom right.
        1   2   3
        4   5   6
        7   8   9
        10  11  12
        :param verbose: Activate debug windows to show what's happening here
        :return: No return
        """

        # check if correct number of segments were detected,
        # exits if number of segments != 12, since this is the basis for the
        # computations in this function
        self.assert_nmbr_segments()

        sorted_segs = sorted(self.segments, key=attrgetter('y'))
        temp_list = []
        for i in range(0, len(sorted_segs), 3):
            row_segs = [sorted_segs[i], sorted_segs[i + 1], sorted_segs[i + 2]]
            row_segs = sorted(row_segs, key=attrgetter('x'))
            row_segs[0].set_id(i + 1)
            row_segs[1].set_id(i + 2)
            row_segs[2].set_id(i + 3)
            # size_updater_for_contour_detection(row_segs, verbose=verbose)

            temp_list.append(row_segs)

        # update segments with correct order and sizes
        self.segments = [item for sublist in temp_list for item in sublist]

        if verbose:
            for seg in self.segments:
                draw_circle(self.rgb_img, seg.x, seg.y, thickness=50)
                cv.rectangle(self.rgb_img, (seg.x, seg.y),
                             (seg.x + seg.w, seg.y + seg.h),
                             consts.YELLOW, 6)
                self.show_img("Sorting Segments", self.rgb_img, divisor=4)
                key = cv.waitKey()
                if "c" == chr(key & 255):
                    continue
                if "q" == chr(key & 255):
                    log.info("Breaking from segment counter")
                    break

    @timeit()
    def process_segments_dbscan(self, seg, verbose=False):
        """Process detected segments
        This method processes the detected segments using morphological
        operations, multiotsu thresholding and DBSCAN.

        :param seg: the segment that is being processed
        :param verbose: Activate debug windows to show what's happening here
        :return: No return
        """
        log.debug("SEG ID = %i" % seg.id)
        self.gray_seg = self.gray_img[seg.y:seg.y + seg.h,
                                      seg.x:seg.x + seg.w]

        if verbose:
            self.rgb_seg = cv.cvtColor(self.gray_seg, cv.COLOR_GRAY2BGR)
            self.show_img("0: Input segment", self.gray_seg, divisor=6)

        # remove reflections
        min_val_pixel = np.min(self.gray_seg)
        self.gray_seg[self.gray_seg == 255] = min_val_pixel
        if verbose:
            self.show_img("1: Removed reflections", self.gray_seg,
                          divisor=6)

        # median blur
        processed_seg = cv.medianBlur(self.gray_seg, self.seg_median_blur)
        if verbose:
            self.show_img("2: Blurred segment", processed_seg, divisor=6)

        # remove "salt" noise
        opening_kernel = cv.getStructuringElement(
            cv.MORPH_ELLIPSE,
            (self.seg_opening_kern_size, self.seg_opening_kern_size))
        processed_seg = cv.morphologyEx(processed_seg, cv.MORPH_OPEN,
                                        opening_kernel,
                                        iterations=self.seg_opening_iters)
        if verbose:
            self.show_img("3: Opening after input", processed_seg,
                          divisor=6)

        # multi otsu thresholding
        thresholds = threshold_multiotsu(processed_seg,
                                         classes=self.nb_otsu_classes)
        processed_seg = np.uint8(np.digitize(processed_seg,
                                             bins=thresholds))
        processed_seg[processed_seg == 0] = 255
        processed_seg[processed_seg == 1] = 150
        processed_seg[processed_seg == 2] = 100
        processed_seg[processed_seg == 3] = 50
        processed_seg[processed_seg == 4] = 0
        processed_seg[processed_seg == 5] = 0
        if verbose:
            self.show_img("5: Multi Otsu", processed_seg, divisor=6)

        # dilate with radial kernel
        radial_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # processed_seg = cv.dilate(processed_seg, radial_kernel,
        #                           iterations=1)
        processed_seg = cv.morphologyEx(processed_seg, cv.MORPH_OPEN,
                                        radial_kernel, iterations=3)
        if verbose:
            self.show_img("6: Openend output", processed_seg, divisor=6)

        closing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        processed_seg = cv.morphologyEx(processed_seg, cv.MORPH_CLOSE,
                                        closing_kernel, iterations=4)
        if verbose:
            self.show_img("7: closing after otsu", processed_seg, divisor=6)

        # Blob detection
        blob_keypts = self.blob_detector_seg.detect(processed_seg)

        # DBSCAN
        keypt_list = []
        for keypt in blob_keypts:
            # print(keypt.size)
            keypt_list.append([keypt.pt[0], keypt.pt[1]])
        dbscan_input = np.array(keypt_list)
        do_dbscan = True
        log.debug(" detected %i blobs" % len(blob_keypts))
        detected_blobs = len(blob_keypts)
        if detected_blobs < 10:
            do_dbscan = False
            log.error("Less than 10 diamonds detected. Is the input "
                      "correct?")
        elif detected_blobs > 100:
            dbscan_eps = 65
        elif detected_blobs > 55:
            dbscan_eps = 75
        else:
            dbscan_eps = 85

        if do_dbscan:
            clustering = DBSCAN(eps=dbscan_eps,
                                min_samples=3).fit(dbscan_input)
            counted_vals = Counter(clustering.labels_)
            log.debug("DBSCAN counted_vals = %s" % str(counted_vals))
            most_frequent = counted_vals.most_common(1)[0]

            for i, keypt_label in enumerate(clustering.labels_):
                if keypt_label == most_frequent[0]:
                    seg.diamonds.append(Blob(
                        int(blob_keypts[i].pt[0] + seg.x),
                        int(blob_keypts[i].pt[1] + seg.y),
                        seg.id))

                if verbose:
                    if keypt_label != most_frequent[0]:
                        draw_circle(self.rgb_img,
                                    int(blob_keypts[i].pt[0] + seg.x),
                                    int(blob_keypts[i].pt[1] + seg.y),
                                    10, consts.RED)
                    else:
                        draw_circle(self.rgb_img,
                                    int(blob_keypts[i].pt[0] + seg.x),
                                    int(blob_keypts[i].pt[1] + seg.y),
                                    10, consts.GREEN)
        else:
            for keypt in blob_keypts:
                seg.diamonds.append(Blob(
                    int(keypt.pt[0] + seg.x),
                    int(keypt.pt[1] + seg.y),
                    seg.id))

        if verbose:
            self.show_img("Current detection", self.rgb_img, divisor=5)
            cv.waitKey()

    @timeit()
    def filter_outliers_top(self, seg, y_range, threshold, verbose=False):
        """ Filters outliers that end up on top of the segments

        :param y_range: Range of pixels in y direction
        :param threshold: threshold for amount of diamonds in the same height
        to say if current diamond is a diamond or outlier
        :param verbose: shows removed outliers in self.rgb_img
        :return: No return
        """
        seg.diamonds.sort(key=attrgetter('y'))
        nb_removed = 0
        # for i, dia in enumerate(seg.diamonds[:10]):
        for i, dia in enumerate(seg.diamonds[:10]):
            in_range = 0
            for other_dia in seg.diamonds:
                y_dist = abs(dia.y - other_dia.y)
                if y_dist <= y_range:
                    in_range += 1
                if in_range > threshold:
                    break
            if in_range <= threshold:
                if verbose:
                    draw_circle(self.rgb_img,
                                seg.diamonds[i - nb_removed].x,
                                seg.diamonds[i - nb_removed].y,
                                20, consts.YELLOW)

                seg.diamonds.pop(i - nb_removed)
                nb_removed += 1
        # if verbose:
        #     cv.waitKey()

    def filter_sides(self, seg, x_range, threshold, verbose):
        nb_removed = 0
        # for i, dia in enumerate(seg.diamonds[:10]):
        for i, dia in enumerate(seg.diamonds[:5]):
            in_range = 0
            for other_dia in seg.diamonds:
                x_dist = abs(dia.x - other_dia.x)
                if x_dist <= x_range:
                    in_range += 1
                if in_range > threshold:
                    break
            if in_range <= threshold:
                if verbose:
                    draw_circle(self.rgb_img,
                                seg.diamonds[i - nb_removed].x,
                                seg.diamonds[i - nb_removed].y,
                                20, consts.YELLOW)

                seg.diamonds.pop(i - nb_removed)
                nb_removed += 1
        # if verbose:
        #     cv.waitKey()

    @timeit()
    def filter_outliers_left_right(self, seg, x_range, threshold,
                                   verbose=False):
        """ Filters outliers that end up left and right of the segments

        :param x_range: Range of pixels in x direction
        :param threshold: threshold for amount of diamonds in the same height
        to say if current diamond is a diamond or outlier
        :param verbose: shows removed outliers in self.rgb_img
        :return: No return
        """
        seg.diamonds.sort(key=attrgetter('x'))
        self.filter_sides(seg, x_range=x_range, threshold=threshold,
                          verbose=verbose)

        seg.diamonds.sort(key=attrgetter('x'), reverse=True)
        self.filter_sides(seg, x_range=x_range, threshold=threshold,
                          verbose=verbose)

    def draw_detected_diamonds(self, segment):
        """Diamond drawer
        This method draws diamond positions for a specific segment on the rgb
        image. For debugging purposes
        :param segment: input segment you want to draw the diamonds for
        :return: No return
        """
        for blob in segment.diamonds:
            draw_circle(self.rgb_img, blob.x, blob.y, thickness=5)

    def run_pipeline(self, file_path, show_io=False):
        """Run pipeline
        This method is used by the backend to invoke the cv pipeline and
        process one image taken by the Basler camera system.

        :param file_path: Path to the file in the filesystem
        :param show_io: For debugging purposes you can show the input and
        output
        :return: Returns segments for the evaluation using labeled data
        """

        # reset image, segments and variables
        self.reset_vars()

        # load new image
        self.set_img(file_path)
        if show_io:
            self.rgb_img = cv.cvtColor(self.gray_img, cv.COLOR_GRAY2BGR)
            self.show_img("Input Image", self.rgb_img, divisor=4)

        # 1. Multi-scale template matching
        self.set_dia_segs_multi_scale_tm(verbose=False)

        # 1.2 Filter segments for more robustness
        self.filter_segments(verbose=False)

        # 2. Sort segments
        # NOTE: This only works if 12 segments are detected.
        self.sort_segments(verbose=False)

        for seg in self.segments:
            # 3. segment processing
            self.process_segments_dbscan(seg, verbose=False)
            # 4. outliers top
            self.filter_outliers_top(seg, y_range=25, threshold=4,
                                     verbose=False)
            # 5. outliers left right
            self.filter_outliers_left_right(seg, x_range=30, threshold=1,
                                            verbose=False)

        # show detected diamonds
        if show_io:
            # draws detected diamonds on input image for frontend
            for seg in self.segments:
                # self.draw_detected_diamonds(seg)
                for diamond in seg.diamonds:
                    draw_circle(self.rgb_img, diamond.x, diamond.y, 5,
                                consts.GREEN)
            self.show_img("Final Output", self.rgb_img, divisor=4)

        return self.segments
