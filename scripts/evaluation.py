import logging
import numpy as np
import cv2 as cv

import hilti.constants as consts
from hilti.image import Blob
from hilti.utilities import scale_bounding_box, draw_circle


log = logging.getLogger(__name__)


class EvaluationSegment:
    def __init__(self, id, x, y, w, h, diamonds):
        self.id = id
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.diamonds = diamonds


class Evaluator:
    def __init__(self):
        # blob detector
        params_blob_seg = cv.SimpleBlobDetector_Params()
        params_blob_seg.blobColor = consts.MAX_PIXEL_VAL
        params_blob_seg.filterByInertia = False
        self.blob_detector = cv.SimpleBlobDetector_create(params_blob_seg)
        log.debug("Evaluator was initialized")
        self.eval_img = np.zeros((3000, 4096), dtype=np.uint8)
        self.eval_segments = []

    def show_img(self, name, src, divisor):
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        height = 3000
        width = 4096
        cv.resizeWindow(name, width, height)
        cv.imshow(name, src)

    def set_true_segments(self, file_path, segments, verbose=False):
        """
        sets
        :param file_path:
        :param segments:
        :param verbose:
        :return:
        """
        split_path = file_path.split('.')
        file_path_no_format = split_path[-2]
        new_file_path = file_path_no_format + "_labeled.bmp"
        log.debug("new file path = %s" % new_file_path)
        # new_file_path =
        self.eval_img = cv.imread(new_file_path, cv.IMREAD_COLOR)
        self.test_img = self.eval_img.copy()

        # get red channel
        red_channel = self.eval_img[:, :, 2].copy()

        # remove background
        red_channel[red_channel != 236] = 0

        # diamonds --> white
        red_channel[red_channel != 0] = 255

        # dilate diamonds
        radial_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        red_channel = cv.dilate(red_channel, radial_kernel, iterations=3)

        blob_keypts = self.blob_detector.detect(red_channel)
        log.debug("%i diamonds" % len(blob_keypts))

        for i, seg in enumerate(segments):
            x, y, w, h = scale_bounding_box(seg.x, seg.y, seg.w, seg.h, 1.2)

            curr_seg = red_channel[y:y + h, x:x + w]
            # if verbose:
            #     cv.rectangle(self.eval_img, (x, y),(x + w, y + h),
            #                  consts.YELLOW, 10)
            seg_keypts = self.blob_detector.detect(curr_seg)
            diamonds = []
            for keypt in seg_keypts:
                diamonds.append(Blob(int(keypt.pt[0]) + x,
                                     int(keypt.pt[1]) + y))

            self.eval_segments.append(EvaluationSegment(i + 1, x, y, w, h,
                                                        diamonds))

        sum_diamonds = 0
        for eval_seg in self.eval_segments:
            sum_diamonds += len(eval_seg.diamonds)
        log.debug("%i diamonds in all segments" % sum_diamonds)

        try:
            assert(len(blob_keypts) == sum_diamonds)
        except AssertionError:
            log.error("Evaluator counts different amount of diamonds in "
                      "total segment and summed per segemnt check if segment"
                      "detection correct by setting verbose=True on "
                      "Evaluator.")

        if verbose:
            for seg in segments:
                x, y, w, h = scale_bounding_box(seg.x, seg.y, seg.w, seg.h,
                                                1.2)
                cv.rectangle(red_channel, (x, y),
                             (x + w, y + h), (255, 255, 255), 10)
            red_channel = \
                cv.drawKeypoints(red_channel, blob_keypts, np.array([]),
                                 consts.GREEN,
                                 cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.show_img("eval img", self.eval_img, divisor=5)
            self.show_img("red channel", red_channel, divisor=5)

    def compare_segments(self, detected_segs, nb, verbose=False):
        """
        This function compares the detected diamonds with the true position of
        the diamonds
        :param detected_segs: segments detected using cv-pipeline
        :param verbose on/off results in OpenCV GUI
        :return:
        """
        # distance threshold between true and detected diamonds in pixels
        dist_th = 20

        # 1. total error
        sum_true_diamonds = 0
        for seg in self.eval_segments:
            sum_true_diamonds += len(seg.diamonds)
        sum_detect_diamonds = 0
        for seg in detected_segs:
            sum_detect_diamonds += len(seg.diamonds)
        total_error = \
            np.abs(sum_true_diamonds - sum_detect_diamonds) / sum_true_diamonds
        log.info(" TOTAL: %i / %i [DETECT / TRUE]: %.1f%% error"
                 % (sum_detect_diamonds, sum_true_diamonds, total_error * 100))

        # 2. error per segment
        total_trefferquote = 0
        nb_total_trefferquote = 0
        total_fehlerrate = 0
        nb_total_fehlerrate = 0
        for true_seg, det_seg in zip(self.eval_segments, detected_segs):
            error_per_seg = \
                np.abs(len(true_seg.diamonds) - len(det_seg.diamonds)) / \
                len(true_seg.diamonds)

            # 3. compute correctly detected per segment
            correct = []
            incorrect = []
            matching_closest = -1
            for i, det_dia in enumerate(det_seg.diamonds):
                min_dist = np.inf
                curr_diamond = np.array([det_dia.x, det_dia.y])

                for j, tru_dia in enumerate(true_seg.diamonds):
                    dist = \
                        np.linalg.norm(curr_diamond - np.array([tru_dia.x,
                                                                tru_dia.y]))
                    if dist < min_dist:
                        min_dist = dist
                        matching_closest = j
                if verbose:
                    draw_circle(self.eval_img, det_dia.x, det_dia.y, 10,
                                consts.BLUE)

                if min_dist < dist_th:
                    correct.append(i)
                    if verbose:
                        draw_circle(self.test_img,
                                    true_seg.diamonds[matching_closest].x,
                                    true_seg.diamonds[matching_closest].y,
                                    10, consts.GREEN)
                else:
                    incorrect.append(i)
                    if verbose:
                        draw_circle(self.eval_img, det_dia.x, det_dia.y,
                                    4, consts.YELLOW)

            accuracy = len(correct) / len(true_seg.diamonds)
            total_trefferquote += accuracy
            nb_total_trefferquote += len(correct)

            if verbose:
                for corr in correct:
                    draw_circle(self.eval_img,
                                det_seg.diamonds[corr].x,
                                det_seg.diamonds[corr].y,
                                4, consts.GREEN)

            # 4. detect falsely detected DIA
            error_falsely = \
                (len(det_seg.diamonds) - len(correct)) / len(true_seg.diamonds)

            # 5. not detected diamonds
            not_found = []
            for i, tru_dia in enumerate(true_seg.diamonds):
                min_dist = np.inf
                curr_diamond = np.array([tru_dia.x, tru_dia.y])
                for j, det_dia in enumerate(det_seg.diamonds):
                    dist = \
                        np.linalg.norm(curr_diamond - np.array([det_dia.x,
                                                                det_dia.y]))
                    if dist < min_dist:
                        min_dist = dist
                if min_dist > dist_th:
                    not_found.append(i)

            error_not_found = len(not_found) / len(true_seg.diamonds)

            if verbose:
                for index_nf in not_found:
                    draw_circle(self.eval_img,
                                true_seg.diamonds[index_nf].x,
                                true_seg.diamonds[index_nf].y,
                                4, consts.YELLOW)
                    self.show_img("eval with diamonds", self.eval_img,
                                  divisor=4)

            nb_fehlerrate = \
                (len(det_seg.diamonds) - len(correct)) + len(not_found)
            nb_total_fehlerrate += nb_fehlerrate
            error_fehlerrate = nb_fehlerrate / len(true_seg.diamonds)
            total_fehlerrate += error_fehlerrate

            # log eval per segment
            log.debug("\tSEG %2i, %i / %i [DETECT / TRUE]: %.1f%% error" %
                      (true_seg.id,
                       len(det_seg.diamonds),
                       len(true_seg.diamonds),
                       error_per_seg * 100))
            log.debug("\t--> TREFFERQUOTE %i / %i [CORRECT / TRUE]: %.1f%% "
                      "accuracy" % (len(correct),
                                    len(true_seg.diamonds),
                                    accuracy * 100))
            log.debug("\t--> FEHLERRATE %i / %i [FR / TRUE]: %.1f%% "
                      "error" % (nb_fehlerrate,
                                 len(true_seg.diamonds),
                                 error_fehlerrate * 100))
            log.debug("\t   --> FALSELY %i / %i [FALSE / TRUE]: %.1f%% "
                      "error" % (len(det_seg.diamonds) - len(correct),
                                 len(true_seg.diamonds),
                                 error_falsely * 100))
            log.debug("\t   --> NOT FOUND %i / %i [NOT FOUND / TRUE]: %.1f%% "
                      "error\n" % (len(not_found),
                                   len(true_seg.diamonds),
                                   error_not_found * 100))
        # log total
        log.info(" TOTAL TREFFERQUOTE (HIT RATIO) %i / %i [HR / TRUE]: %.1f%% "
                 "accuracy" % (nb_total_trefferquote,
                               sum_true_diamonds,
                               (total_trefferquote / 12) * 100))
        log.info(" TOTAL FEHLERRATE (ERROR RATE) %i / %i [FR / TRUE]: %.1f%% "
                 "error" % (nb_total_fehlerrate,
                            sum_true_diamonds,
                            (total_fehlerrate / 12) * 100))

        return total_trefferquote, total_fehlerrate

    def reset_vars(self):
        self.eval_segments = []
