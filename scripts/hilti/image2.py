import cv2 as cv
import numpy as np
import logging
from matplotlib import pyplot as plt # Debug
import os # Debg
from skimage.filters import threshold_multiotsu

from hilti.segment import Segment
from hilti.blob import Blob
import hilti.constants as consts
from hilti.utilities import draw_circle
from hilti.utilities import show_img
from hilti.utilities import timeit

log = logging.getLogger(__name__)


def adjust_bottom(processed_seg, seg, verbose =  False):
    print("Bottom   ")
    black_count = 0
    margin_b = -5
    i = 0
    while black_count < 90:
        #print("margin_b ", margin_b, 'i ' , i)
        i+=1
        if black_count == 0:
            margin_b += 10
        else:
            margin_b += 5

        if verbose and margin_b>0: 
            #print("i  ", i)
            name_wondiow = "Bottom Margin {}".format(i)
            bottom_seg = processed_seg[processed_seg.shape[0]-margin_b:, :]
            show_img(name_wondiow, bottom_seg, divisor=1)
        bottom_seg = processed_seg[processed_seg.shape[0]-margin_b:, :]
        black_count = np.array(bottom_seg ==0).sum()
        print("BOTTOM black_count  ", black_count)
        

    print("margin_b  ", margin_b)   
    margin_b -= 5
    print("margin_b  RECTIF ", margin_b)  
    seg.adjust_bottom_border(margin_b)
    print("bottom seg ",seg.bottom_border )
    processed_seg[processed_seg.shape[0]-margin_b:processed_seg.shape[0], :] = 255
    return processed_seg

def adjust_right(processed_seg, seg, verbose = False):
    black_count = 0
    margin_r = -10
    i =0
    while black_count < 90:
        i+=1
        if black_count == 0:
            margin_r += 10
        else:
            margin_r += 5
        right_seg = processed_seg[20:,processed_seg.shape[1]-margin_r:]
        black_count = np.array(right_seg ==0).sum()
        print(" Black count R", black_count)
        if verbose and margin_r>0: 
            name_wondiow = "Right Margin{}".format(i)
            right_seg = processed_seg[20:,processed_seg.shape[1]-margin_r:]
            show_img(name_wondiow, right_seg, divisor=1)
    print("margin_r  ", margin_r,processed_seg.shape[1] - (processed_seg.shape[1]-margin_r) )  
    margin_r -= 15
    print("margin_r RECTIF  ", margin_r,processed_seg.shape[1] - (processed_seg.shape[1]-margin_r)  )
    seg.adjust_right_border(margin_r)
    processed_seg[ :,processed_seg.shape[1]-margin_r:] = 255
    return processed_seg

def adjust_left(processed_seg, seg, verbose = False):
    print(" SEG ID ", seg.id)
    black_count = 0
    margin_l = 10
    i =0
    while black_count < 90:
        i += 1
        if black_count == 0:
            margin_l += 10
        else:
            margin_l += 5
        print(" margin_l ", margin_l,  " i  ",  i, " SEG ID ", seg.id)
        if seg.id != 10 :
            left_seg = processed_seg[35:,10:margin_l]
        else:
            left_seg = processed_seg[55:,10:margin_l]
        black_count = np.array(left_seg ==0).sum()
        print(" Black count ", black_count)

        if verbose and margin_l>10: 
                name_wondiow = "Left Margin{} , segID {}".format(i, seg.id)
                left_seg = processed_seg[35:,10:margin_l]
                show_img(name_wondiow, left_seg, divisor=1)
    print("margin_l  F before rectif ", margin_l)   
    margin_l -= 5
    seg.adjust_left_border(margin_l)
    processed_seg[:,:margin_l] = 255
    return processed_seg


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
            params.minDistBetweenBlobs = 5 #10
            params.minRepeatability = 1  # default = 2

            params.minThreshold = self.bd_minThreshold
            params.maxThreshold = self.bd_maxThreshold
            params.thresholdStep = self.bd_thresholdStep  # default = 10
            # params.blobColor = consts.MAX_PIXEL_VAL
            params.filterByColor = True
            params.blobColor = 0 # Black 

            params.filterByArea = True
            params.minArea = 60.
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
                [553, 541, 1398, 786], #[560, 560, 1398, 791]
                # segment 2
                [1583, 530, 2456, 764], #[1580, 549, 2456, 779]
                # segment 3
                [2670, 513, 3488, 756], #[2637, 527, 3488, 761] #3500
                # segment 4
                [493, 969, 1379, 1227],#[500, 988, 1379, 1232],
                # segment 5
                [1576, 956, 2460, 1211], #[1573, 975, 2460, 1219]
                # segment 6
                [2653, 944, 3544, 1214], #[2661, 963, 3544, 1204]
                # segment 7
                [428, 1435, 1360, 1718], #[435, 1454, 1360, 1713]
                # segment 8
                [1553, 1411, 2487, 1707], #[1550, 1430, 2487, 1713]
                # segment 9
                [2686, 1404, 3608, 1685], #[2683, 1420, 3608, 1690]
                # segment 10
                [359, 1924, 1316, 2236], #[366, 1943, 1316, 2261],
                # segment 11
                [1532, 1918, 2498, 2236], #[1529, 1937, 2498, 2236],
                # segment 12
                [2802, 1908, 3678, 2214] #[2709, 1927, 3678, 2229]
            ]
        else:
            # for polygons x,y coordinates for
            # top left, bottom left, bottom right, top right (in this order)
            pts_list = [
                # segment 1
                np.array(([553, 541], [553, 786], [1398, 786], [1398, 541]),
                         dtype=np.int32),
                # segment 2
                np.array(([1583, 530], [1583, 774], [2456, 774], [2456, 526]),
                         dtype=np.int32),
                # segments 3
                np.array(([2640, 513], [2640, 756], [3488, 756], [3488, 513]),
                         dtype=np.int32),
                # segments 4
                np.array(([493, 969], [493, 1227], [1379, 1227], [1379, 969]),
                         dtype=np.int32),
                # segments 5
                np.array(([1576, 956], [1576, 1219], [2460, 1219], [2460,
                                                                    1194]),
                         dtype=np.int32),
                # segments 6
                np.array(([2654, 947], [2654, 1214], [3544, 1214], [3544,
                                                                    947]),
                         dtype=np.int32),
                # segments 7
                np.array(([428, 1435], [428, 1718], [1360, 1718], [1360,
                                                                   1435]),
                         dtype=np.int32),
                # segments 8
                np.array(([1553, 1414], [1553, 1707], [2487, 1707], [2487,
                                                                     1414]),
                         dtype=np.int32),
                # segments 9
                np.array(([2686, 1404], [2686, 1695], [3608, 1695], [3608,
                                                                     1404]),
                         dtype=np.int32),
                # segments 10
                np.array(([359, 1924], [359, 2246], [1316, 2246], [1316,
                                                                   2231]),
                         dtype=np.int32),
                # segments 11
                np.array(([1532, 1918], [1532, 2241], [2498, 2236], [2498,
                                                                     2236]),
                         dtype=np.int32),
                # segments 12
                np.array(([2712, 1908], [2712, 2214], [3678, 2214], [3678,
                                                                     2209]),
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

    def draw_upper_demarcation(self):
        # # x1, y1, x2, y2
        #         # segment 1
        #         [553, 541, 1398, 786], #[560, 560, 1398, 791]
        #         # segment 2
        #         [1583, 530, 2456, 764], #[1580, 549, 2456, 779]
        #         # segment 3
        #         [2670, 513, 3488, 756], #[2637, 527, 3488, 761] #3500
        #         # segment 4
        #         [493, 969, 1379, 1227],#[500, 988, 1379, 1232],
        #         # segment 5
        #         [1576, 956, 2460, 1211], #[1573, 975, 2460, 1219]
        #         # segment 6
        #         [2653, 944, 3544, 1214], #[2661, 963, 3544, 1204]
        #         # segment 7
        #         [428, 1435, 1360, 1718], #[435, 1454, 1360, 1713]
        #         # segment 8
        #         [1553, 1411, 2487, 1707], #[1550, 1430, 2487, 1713]
        #         # segment 9
        #         [2686, 1404, 3608, 1685], #[2683, 1420, 3608, 1690]
        #         # segment 10
        #         [359, 1924, 1316, 2236], #[366, 1943, 1316, 2261],
        #         # segment 11
        #         [1532, 1918, 2498, 2236], #[1529, 1937, 2498, 2236],
        #         # segment 12
        #         [2802, 1908, 3678, 2214] #[2709, 1927, 3678, 2229]

        # Line Seg 1:
        self.gray_img[560-3:560,560:1398 ]=79 
        # Line Seg 2:
        self.gray_img[549-3:549, 1580:2456]=79 
        # Line Seg 3:
        self.gray_img[549-3:549, 2637:3488]=79 
        # Line Seg 4:
        self.gray_img[988-3:988, 988:1232]=79 
        # Line Seg 5:
        self.gray_img[975-3:975, 1573:2460]=79 
        # Line Seg 6:
        self.gray_img[1454-3:1454, 2661:3544]=79 
        # Line Seg 7:
        self.gray_img[435-3:435, 435:1360]=79 
        # Line Seg 8:
        self.gray_img[1430-3:1430, 1550:2487]=79 
        # Line Seg 9:
        self.gray_img[1420-3:1420, 2683:3608]=79 
        # Line Seg 10:
        self.gray_img[1943-3:1943, 366:1316]=79 
        # Line Seg 11:
        self.gray_img[1937-3:1937, 1529:2498]=79 
        # Line Seg 12:
        self.gray_img[1927-3:1927:, 2709:3678]=79 
        
        
                
    @timeit()
    def process_segments(self, seg, inner_seg=None, verbose=False, verbose_seg=False):
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

        # f_gray_seg = self.gray_seg.flatten()
        # plt.hist(f_gray_seg, bins =list(range(np.amin(f_gray_seg),np.amax(f_gray_seg),2 ))) 
        # plt.title("histogram") 
        # plt.savefig("hist.png")
        print("\n \n MIN ", np.amin(self.gray_seg), " MAX ", np.amax(self.gray_seg))
        if self.use_polygons:
            # remove reflections
            min_val_pixel = np.min(self.gray_seg)
            # self.gray_seg[self.gray_seg == 255] = min_val_pixel
            if min_val_pixel > 0:
                self.gray_seg[self.gray_seg >= 120] = min_val_pixel

            else:
                self.gray_seg[self.gray_seg >= 120] = 1

            if min_val_pixel > 0:
                upper_gray = self.gray_seg[:round(self.gray_seg.shape[0]/2),:]
                upper_gray[upper_gray >= 80] = min_val_pixel
                self.gray_seg[:round(self.gray_seg.shape[0]/2), :] = upper_gray

            else:
                upper_gray = self.gray_seg[:round(self.gray_seg.shape[0]/2)]
                upper_gray[upper_gray >= 80 ] = 1
                self.gray_seg[:round(self.gray_seg.shape[0]/2), :] = upper_gray


            if min_val_pixel > 0:
                gray_23 = self.gray_seg[round(self.gray_seg.shape[0]/2):round((self.gray_seg.shape[0]/3)*2),:]
                gray_23[gray_23 >= 100] = min_val_pixel
                self.gray_seg[round(self.gray_seg.shape[0]/2):round((self.gray_seg.shape[0]/3)*2),:] = gray_23

            else:
                gray_23 = self.gray_seg[round(self.gray_seg.shape[0]/2):round((self.gray_seg.shape[0]/3)*2),:]
                gray_23[gray_23 >= 100] = 1
                self.gray_seg[round(self.gray_seg.shape[0]/2):round((self.gray_seg.shape[0]/3)*2),:] = gray_23
            


            #show_img("00 gray_seg remove white pixel", self.gray_seg, 1)
            # inner_seg.pts -= inner_seg.pts.min(axis=0)
            # mask = np.zeros(self.gray_seg.shape[:2], dtype=np.uint8)
            # show_img("mask", mask, 1)
            # cv.drawContours(mask, [inner_seg.pts], -1, (255, 255, 255), -1,
            #                 cv.LINE_AA)
            # self.gray_seg[self.gray_seg >= 250] = 1
            # self.gray_seg = cv.bitwise_and(self.gray_seg, self.gray_seg,
            #                                mask=mask)
            # show_img("0 gray_seg", self.gray_seg, 1)
            # # self.gray_seg[self.gray_seg == 0] = 255

            # bg = np.ones_like(self.gray_seg, dtype=np.uint8) * 255
            # cv.bitwise_not(bg, bg, mask=mask)
            # self.gray_seg = cv.bitwise_xor(self.gray_seg, bg)
            # show_img("1 gray_seg", self.gray_seg, 1)

            # show_img("bg", bg, 1)
            #show_img("1 OUTER gray_seg", self.gray_seg, 1)
            # self.gray_seg += bg

            #
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
        #processed_seg = cv.medianBlur(self.gray_seg, 11)

        processed_seg = cv.medianBlur(self.gray_seg, 11)
        upper_upper_blur = processed_seg[:12,:] 
        upper_upper_blur =  cv.medianBlur(upper_upper_blur, 25)
        processed_seg[0:12, :] = upper_upper_blur

        upper_blur = processed_seg[:12,:] 
        upper_upper_blur =  cv.medianBlur(upper_upper_blur, 25)
        processed_seg[0:12, :] = upper_upper_blur
        #cv.rectangle(processed_seg, (0 ,0),
        #                     ( processed_seg.shape[1], 12 ),
        #                    consts.YELLOW, thickness=1)
        upper_blur = processed_seg[12:30, :] 
        upper_blur =  cv.medianBlur(upper_blur, 3)
        processed_seg[12:30, :]  = upper_blur


        # cv.rectangle(processed_seg, (0,12),
        #                      (processed_seg.shape[1] , 60),
        #                     consts.RED, thickness=1)
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
        thresholds = threshold_multiotsu(processed_seg, classes=5)

        #thresholds2 = threshold_multiotsu(processed_seg, classes=6)
        print("\n \n thresholds   :  ", thresholds, thresholds)
        # thresholds[0] -= 3
        processed_seg = np.uint8(np.digitize(processed_seg, bins=thresholds))
        processed_seg[processed_seg == 0] = 0
        processed_seg[processed_seg == 1] = 49
        processed_seg[processed_seg == 2] = 99
        processed_seg[processed_seg == 3] = 255
        processed_seg[processed_seg == 4] = 255
        processed_seg[processed_seg == 5] = 255
        processed_seg[processed_seg == 6] = 255  

        processed_seg = adjust_right(processed_seg, seg, False)
        processed_seg = adjust_left(processed_seg, seg, False)
        processed_seg = adjust_bottom(processed_seg, seg, False)
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

        if verbose_seg:
           
            print("SEG INFO / ", seg.id, seg.x, seg.y, seg.left_border, seg.right_border)
            cv.rectangle(self.rgb_img, (seg.x + seg.left_border , seg.y),
                             (seg.x + seg.w - seg.right_border, seg.y + seg.h ),
                            consts.BLUE, thickness=1)

        if verbose:
            for diamond in seg.diamonds:
                draw_circle(self.rgb_img, diamond.x, diamond.y, 5,
                            consts.GREEN)

            show_img("Current detection", self.rgb_img, divisor=5)
            cv.waitKey()

 
    def run_pipeline(self, file_path, show_io=False, create_rgb_img=False, save_io = False):
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
        #if show_io:
        #    show_img("Input Image", self.rgb_img, divisor=4)

        self.set_segs_fixed_bounding_boxes(use_polygons=True, verbose=True)
        #self.draw_upper_demarcation()

        
        for seg in self.segments:
            # 3. segment processing
            self.process_segments(seg, verbose=True, verbose_seg =True)
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
                                consts.RED)
            show_img("Final Output", self.rgb_img, divisor=4)
        if save_io:
               for seg in self.segments:
                # self.draw_detected_diamonds(seg)
                for diamond in seg.diamonds:
                    draw_circle(self.rgb_img, diamond.x, diamond.y, 5,
                                consts.RED)

                fname = file_path.split("/")[-1]
                #cv.imwrite("outputs2/"+fname, self.rgb_img)
                #print("\n \n file_path ", file_path)
            #show_img("Final Output", self.rgb_img, divisor=4)

        return self.segments, self.rgb_img
