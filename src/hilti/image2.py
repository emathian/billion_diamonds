import cv2 as cv
import numpy as np
import logging
from skimage.filters import threshold_multiotsu 

from hilti.segment import Segment
from hilti.blob import Blob
import hilti.constants as consts
from hilti.utilities import draw_circle
from hilti.utilities import show_img
from hilti.utilities import timeit

log = logging.getLogger(__name__)


class Image:
    def __init__(self, use_polygons=False, border=0):
        self.rgb_img = None
        self.gray_img = None
        self.segments = []
        self.orig_height = 3000
        self.orig_width = 4096
        # self.mean_pixels, self.std_pixels = cv.meanStdDev(self.gray_img)

        # process segment variables
        self.seg_median_blur = 9
        self.seg_opening_kern_size = 7
        self.seg_opening_iters = 3
        self.nb_otsu_classes = 5

        # blob detector segments
        self._blob_detector_seg = None
        self.bd_minThreshold = 0
        self.bd_maxThreshold = 110
        self.bd_thresholdStep = 50

        # polygons
        self.use_polygons = use_polygons
        self.border = border

    def reset_vars(self):
        """Reset variables
        This method resets the variables needed to process a new image.
        :return: No return
        """
        self.rgb_img = None
        self.gray_img = None
        self.segments = []

    def set_img(self, img_path, resize_divisor=1):
        """Set new image for cv-pipeline

        :param img_path: The path to the next image that will be processed
        :param resize_divisor: If image was resized set it to the corresponding
        value it was resized with
        :param verbose: Converts the input image to BGR for debugging purposes
        :return: No return
        """
        self.gray_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)

    @property
    def blob_detector_seg(self):
        if self._blob_detector_seg is None:
            params = cv.SimpleBlobDetector_Params()
            params.minDistBetweenBlobs = 10
            params.minRepeatability = 1  # default = 2

            params.minThreshold = self.bd_minThreshold
            params.maxThreshold = self.bd_maxThreshold
            params.thresholdStep = self.bd_thresholdStep  # default = 10
            # params.blobColor = consts.MAX_PIXEL_VAL
            params.filterByColor = True
            params.blobColor = 0

            params.filterByArea = True
            params.minArea = 10.
            # params.maxArea = 40
            params.filterByInertia = False

            params.filterByCircularity = False
            # params.minCircularity = .5
            params.filterByConvexity = False
            # params.minConvexity = .75
            self._blob_detector_seg = cv.SimpleBlobDetector_create(params)
        return self._blob_detector_seg

    @timeit()
    def set_segs_fixed_bounding_boxes(self, verbose=False, use_polygons=False):

        if not self.use_polygons:
            pts_list = [
                # x1, y1, x2, y2
                # segment 1
                [560, 560, 1398, 791],
                # segment 2
                [1580, 549, 2456, 779],
                # segment 3
                [2637, 527, 3488, 761],
                # segment 4
                [500, 988, 1379, 1232],
                # segment 5
                [1573, 975, 2460, 1219],
                # segment 6
                [2661, 963, 3544, 1204],
                # segment 7
                [435, 1454, 1360, 1713],
                # segment 8
                [1550, 1430, 2487, 1713],
                # segment 9
                [2683, 1420, 3608, 1690],
                # segment 10
                [366, 1943, 1316, 2261],
                # segment 11
                [1529, 1937, 2498, 2236],
                # segment 12
                [2709, 1927, 3678, 2229]
            ]
        else:
            # for polygons x,y coordinates for
            # top left, bottom left, bottom right, top right (in this order)
            pts_list = [
                # segment 1
                np.array(([561, 562], [524, 809], [1398, 796], [1410, 550]),
                         dtype=np.int32),
                # segment 2
                np.array(([1588, 546], [1577, 789], [2459, 780], [2443, 526]),
                         dtype=np.int32),
                # segments 3
                np.array(([2626, 535], [2638, 780], [3515, 769], [3483, 526]),
                         dtype=np.int32),
                # segments 4
                np.array(([505, 989], [467, 1253], [1374, 1241], [1390, 975]),
                         dtype=np.int32),
                # segments 5
                np.array(([1573, 980], [1561, 1240], [2479, 1228], [2468,
                                                                    960]),
                         dtype=np.int32),
                # segments 6
                np.array(([2651, 961], [2663, 1229], [3580, 1219], [3533,
                                                                    950]),
                         dtype=np.int32),
                # segments 7
                np.array(([442, 1449], [400, 1737], [1349, 1728], [1362,
                                                                   1436]),
                         dtype=np.int32),
                # segments 8
                np.array(([1558, 1437], [1537, 1726], [2499, 1716], [2488,
                                                                     1424]),
                         dtype=np.int32),
                # segments 9
                np.array(([2672, 1447], [2686, 1716], [3637, 1704], [3599,
                                                                     1410]),
                         dtype=np.int32),
                # segments 10
                np.array(([370, 1944], [330, 2257], [1316, 2256], [1322,
                                                                   1938]),
                         dtype=np.int32),
                # segments 11
                np.array(([1526, 1936], [1513, 2253], [2512, 2244], [2501,
                                                                     1925]),
                         dtype=np.int32),
                # segments 12
                np.array(([2696, 1923], [2714, 2243], [3703, 2225], [3668,
                                                                     1909]),
                         dtype=np.int32)
            ]

        # set segments
        for i, pts in enumerate(pts_list):
            if not self.use_polygons:
                self.segments.append(Segment(pts[0], pts[1], pts[2] - pts[0],
                                             pts[3] - pts[1]))
            else:
                x, y, w, h = cv.boundingRect(pts)
                self.segments.append(Segment(x, y, w, h))
                self.segments[-1].set_pts(pts)

            self.segments[-1].set_id(i + 1)

        if verbose:
            for seg in self.segments:
                cv.rectangle(self.rgb_img, (seg.x, seg.y),
                             (seg.x + seg.w, seg.y + seg.h),
                             consts.TEAL, thickness=1)

    @timeit()
    def process_segments(self, seg, verbose=False):
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
        if self.use_polygons:
            # remove reflections
            min_val_pixel = np.min(self.gray_seg)
            # self.gray_seg[self.gray_seg == 255] = min_val_pixel
            if min_val_pixel > 0:
                self.gray_seg[self.gray_seg >= 250] = min_val_pixel
            else:
                self.gray_seg[self.gray_seg >= 250] = 1

            seg.pts -= seg.pts.min(axis=0)
            mask = np.zeros(self.gray_seg.shape[:2], dtype=np.uint8)
            show_img("mask", mask, 1)
            cv.drawContours(mask, [seg.pts], -1, (255, 255, 255), -1,
                            cv.LINE_AA)
            self.gray_seg = cv.bitwise_and(self.gray_seg, self.gray_seg,
                                           mask=mask)
            show_img("gray_seg", self.gray_seg, 1)
            # self.gray_seg[self.gray_seg == 0] = 255

            bg = np.ones_like(self.gray_seg, dtype=np.uint8) * 255
            cv.bitwise_not(bg, bg, mask=mask)
            self.gray_seg = cv.bitwise_xor(self.gray_seg, bg)
            show_img("gray_seg2", self.gray_seg, 1)

            show_img("bg", bg, 1)
            # self.gray_seg += bg

        if verbose:
            self.rgb_seg = cv.cvtColor(self.gray_seg, cv.COLOR_GRAY2BGR)
            show_img("0: Input segment", self.gray_seg, divisor=1)

        # remove reflections
        # # ToDo moved this out of here
        # min_val_pixel = np.min(self.gray_seg)
        # # self.gray_seg[self.gray_seg == 255] = min_val_pixel
        # self.gray_seg[self.gray_seg == 255] = min_val_pixel
        # if verbose:
        #     show_img("1: Removed reflections", self.gray_seg, divisor=1)

        # median blur
        # processed_seg = cv.medianBlur(self.gray_seg, 11)
        processed_seg = cv.medianBlur(self.gray_seg, 11)
        # processed_seg = cv.GaussianBlur(self.gray_seg, (11,11), 5)
        if verbose:
            show_img("2: Blurred segment", processed_seg, divisor=1)

        # # remove "salt" noise
        # opening_kernel = cv.getStructuringElement(
        #     cv.MORPH_ELLIPSE,
        #     (self.seg_opening_kern_size, self.seg_opening_kern_size))
        # processed_seg = cv.morphologyEx(self.gray_seg, cv.MORPH_OPEN,
        #                                 opening_kernel,
        #                                 iterations=3)
        # if verbose:
        #     show_img("3: Opening after input", processed_seg,
        #                   divisor=1)

        # denoise
        cv.fastNlMeansDenoising(processed_seg, processed_seg, h=5,
                                templateWindowSize=7, searchWindowSize=21)

        # test_seg2 = cv.fastNlMeansDenoising(processed_seg, None, h=10,
        #                                         templateWindowSize=5,
        #                                         searchWindowSize=21)
        if verbose:
            show_img("3: Denoised image", processed_seg, divisor=1)
            # show_img("3.1: Denoised image high h", test_seg2, divisor=1)

        # multi otsu thresholding
        print("HERE")
        thresholds = skimage.filters.threshold_multiotsu(processed_seg, classes=5)
        # thresholds[0] -= 3
        processed_seg = np.uint8(np.digitize(processed_seg, bins=thresholds))
        processed_seg[processed_seg == 0] = 0
        processed_seg[processed_seg == 1] = 49
        processed_seg[processed_seg == 2] = 99
        processed_seg[processed_seg == 3] = 255
        processed_seg[processed_seg == 4] = 255
        processed_seg[processed_seg == 5] = 255
        processed_seg[processed_seg == 6] = 255
        if verbose:
            show_img("5: Multi Otsu", processed_seg, divisor=1)

        # # dilate with radial kernel
        # radial_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # # processed_seg = cv.dilate(processed_seg, radial_kernel,
        # #                           iterations=1)
        # processed_seg = cv.morphologyEx(processed_seg, cv.MORPH_OPEN,
        #                                 radial_kernel, iterations=3)
        # if verbose:
        #     show_img("6: Openend output", processed_seg, divisor=1)
        #
        # closing_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        # processed_seg = cv.morphologyEx(processed_seg, cv.MORPH_CLOSE,
        #                                 closing_kernel, iterations=4)
        # if verbose:
        #     show_img("7: closing after otsu", processed_seg, divisor=1)
        if verbose:
            ths = [i for i in range(self.bd_minThreshold,
                                    self.bd_maxThreshold,
                                    self.bd_thresholdStep)]
            for th in ths:
                _, img = cv.threshold(processed_seg, th, 255, cv.THRESH_BINARY)
                show_img("threshold for %s" % str(th), img, divisor=1)

        # # create border for diamonds close to segment edge
        border = self.border  # in pixels
        if border:
            processed_seg = cv.copyMakeBorder(processed_seg, border, border,
                                              border, border,
                                              borderType=cv.BORDER_CONSTANT,
                                              value=255)
            if verbose:
                show_img("7: With border", processed_seg, divisor=1)

        # Blob detection
        blob_keypts = self.blob_detector_seg.detect(processed_seg)
        log.debug(" detected %i blobs" % len(blob_keypts))

        for keypt in blob_keypts:
            seg.diamonds.append(Blob(
                int(keypt.pt[0] - border + seg.x),
                int(keypt.pt[1] - border + seg.y),
                seg.id))

        if verbose:
            for diamond in seg.diamonds:
                draw_circle(self.rgb_img, diamond.x, diamond.y, 5,
                            consts.GREEN)
            show_img("Current detection", self.rgb_img, divisor=5)
            cv.waitKey()

    @timeit()
    def run_pipeline(self, file_path, show_io=False, create_rgb_img=False):

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

        if create_rgb_img:
            self.rgb_img = cv.cvtColor(self.gray_img, cv.COLOR_GRAY2BGR)
        if show_io:
            show_img("Input Image", self.rgb_img, divisor=4)

        self.set_segs_fixed_bounding_boxes(use_polygons=True, verbose=True)

        for seg in self.segments:
            # 3. segment processing
            self.process_segments(seg, verbose=False)
            # # 4. outliers top
            # self.filter_outliers_top(seg, y_range=25, threshold=4,
            #                          verbose=False)
            # # 5. outliers left right
            # self.filter_outliers_left_right(seg, x_range=30, threshold=1,
            #                                 verbose=False)

        # show detected diamonds
        if show_io:
            # draws detected diamonds on input image for frontend
            for seg in self.segments:
                # self.draw_detected_diamonds(seg)
                for diamond in seg.diamonds:
                    draw_circle(self.rgb_img, diamond.x, diamond.y, 5,
                                consts.GREEN)
            show_img("Final Output", self.rgb_img, divisor=4)

        return self.segments
