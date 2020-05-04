from matplotlib import pyplot as plt

import os
import time
import cv2 as cv
import numpy as np
import logging
import functools
from collections import Counter
from itertools import islice

from PIL import Image
from hilti.segment import Segment

import hilti.config as cfg
import hilti.constants as consts


log = logging.getLogger(__name__)


def timeit(default_name='', loglvl=logging.DEBUG):
    """Return a decorator used to report a method's execution time."""
    def decorator(func):
        name = default_name or getattr(func, '__name__', 'Something')
        mesg = '%s .. completed in %.2f sec.'

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            started_at = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                log.log(loglvl, mesg % (name, time.time() - started_at))
        return wrapped

    return decorator


def plot_histogram(src):
    """Plot Image Histogram of OpenCV Mat

    Plots a histogram of a current image using Matplotlib

    :param src: The numpy array containing the data of the OpenCV image
    :return: No return
    """
    plt.hist(src.ravel(), 256, [0, 256])
    plt.show()


def plot_histogram_w_tholds(src, thresholds):
    """Plot Image Histogram of OpenCV Mat with thresholds

    Plots a histogram of a current image using Matplotlib including previously
    computed thresholds

    :param src: The numpy array containing the data of the OpenCV image
    :param thresholds: Python list containing thresholds that will be plotted
    in the histogram
    :return: No Return
    """
    plt.hist(src.ravel(), 256, [0, 256])
    for x in thresholds:
        plt.axvline(x=x, color='k', linestyle='--')
    plt.show()


def plot_hsv_histogram(src):
    """Plot Histogram of image in HSV color model

    :param src: The numpy array containing the data of the OpenCV image
    :return: No return
    """
    hist = cv.calcHist([src], [0, 1], None, [180, 256], [0, 180, 0, 256])
    plt.imshow(hist, interpolation="nearest")
    plt.show()


def get_odd_number(number):
    """Get odd number

    Computes the next higher odd number, returns same if number is already odd

    :param number: The number you want to check and change if odd
    :return number: The updated number (if even it is number + 1)
    """
    if bool(number & 1):
        return number
    else:
        return number + 1


def draw_circle(dst, x, y, thickness, color=consts.RED):
    """Draw circle

    Draws a circle on an OpenCV Mat

    :param dst: The OpenCV Mat you want to draw the circle on
    :param x: x-coordinate of the center of the circle
    :param y: y-coordinate of the center of the circle
    :param thickness: the thickness of the circle
    :param color: the color you want to draw the circle in
    """
    cv.circle(dst, (int(x), int(y)), 1, color, thickness)


def scale_bounding_box(x, y, w, h, scale, up_height=True, up_width=True):
    """Scale bounding box

    Scales a bounding box according to the self.bounding_box_scaling value
    while keeping center fix

    :param x: x-coordinate of rectangle (top-left corner)
    :param y: y-coordinate of rectangle (top-right corner)
    :param w: width of rectangle
    :param h: height of rectangle
    :param scale: scale you want to change the bounding box by
    :param up_height: if true updates height, if false doesn't update height
    :param up_width: if true updates width, if false doesn't update width
    """

    # 1. compute difference in width / height
    dw = int(w * scale) - w
    dh = int(h * scale) - h

    # 2. update x, y, w, h
    # width
    if up_width:
        x -= dw // 2
        w += dw

    # height
    if up_height:
        y -= dh // 2
        h += dh

    return x, y, w, h


def size_updater_for_contour_detection(row_segs):
    """Size updater contour

    Averages the position of a row of bounding boxes based on their computed
    positions.

    :parm row_segs: a list of 3 segments which corresponds to a row of segments
    in the input image
    """
    # average width and height per row
    # 1. width
    # get width of center segment and update left right
    center_width = row_segs[1].w
    # update left segment
    dw = center_width - row_segs[0].w
    row_segs[0].x -= dw
    row_segs[0].w += dw
    # update right segment
    dw = center_width - row_segs[2].w
    row_segs[2].x += dw // 2
    row_segs[2].w += dw // 2

    # update x positions by averaging distance between segments
    dist_x_left = np.abs(row_segs[0].x - row_segs[1].x)
    dist_x_right = np.abs(row_segs[1].x - row_segs[2].x)
    average = (dist_x_left + dist_x_right) // 2
    dx_left = average - dist_x_left
    dx_right = average - dist_x_right
    row_segs[0].x += dx_left
    row_segs[2].x += dx_right

    # 2. height
    # average y of row for all
    # new_y = max(row_segs[0].y, row_segs[1].y, row_segs[2].y)
    min_height = min(row_segs[0].h, row_segs[1].h, row_segs[2].h)
    for i in range(len(row_segs)):
        dh = row_segs[i].h - min_height
        if dh > 0:
            # row_segs[i].y -= dh // 2
            row_segs[i].h -= dh // 2


def mkdir_p(path):
    """Create a directory at `path` mimicing the behavior of `mkdir -p`."""
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def arr_read(path):
    """Return an array object loaded from disk."""
    log.info('Will load arrays from %s', path)
    _, ext = os.path.splitext(path)
    return cv.imread(path) if ext not in ('.npz', '.npy', ) else np.load(path)


def arr_write(path, *arrays, compress=False):
    """Save a NumPy array to disk - optionally compressed."""
    _, ext = os.path.splitext(path)

    assert ext == '.npz' or not compress, 'Only .npz format can be compressed'
    assert ext == '.npz' or len(arrays) == 1, 'Archive supported only by .npz'

    if ext in ('.npz', ):
        savez = np.savez_compressed if compress else np.savez
        savez(path, *arrays)
    elif ext in ('.npy', ):
        np.save(path, arrays[0])
    elif ext in ('.png', '.bmp', ):
        cv.imwrite(path, arrays[0])
    else:
        raise TypeError('Format {ext} not supported')

    log.info('Saved arrays to disk: path=%s', path)


def arr_write_safe(path, *arrays, compress=False):
    """Save a NumPy array to disk safely.

    The `arrays` are firstly written into a temporary file under the same
    directory and then moved to `path` to prevent it from being corrupted.
    The move operation will directly modify the underlying inodes. If the
    temporary file and `path` are not mounted on the same filesystem, the
    move operation will fail and a copy will have to be performed, which
    would beat the purpose of this.

    The directory tree to `path` will also be idempotently created if it
    doesn't already exist.

    """
    _, ext = os.path.splitext(path)
    path_tmp = f'{path}.tmp{ext}'
    mkdir_p(os.path.dirname(path))
    arr_write(path_tmp, *arrays, compress=compress)
    os.rename(path_tmp, path)


def get_path_to_captured_image(batch_id, image_id, ext='png'):
    """Return the path to a captured image on the filesystem.

    NOTE that the `image_id` is NOT an ever increasing number, but rather a
    counter - an integer relative to the batch the image belongs to - which
    resets with each new batch.

    The file extension `.png` is the default, which allows for lossless and
    compressed persistance of images on disk.

    """
    path = f'batch-{batch_id}-image-{image_id}.{ext}'
    return os.path.join(cfg.CAPTURES_DIR, path)


def get_path_to_annotated_image(batch_id, image_id, ext='png'):
    """Return the path of the final image after it's been processed."""
    path = get_path_to_captured_image(batch_id, image_id, ext=ext)
    path, ext = os.path.splitext(path)
    return f'{path}-annotated{ext}'


def get_path_to_extracted_segment(batch_id, image_id, segment_id, ext='png'):
    """Return the path to an extracted segment of an image.

    NOTE that the `segment_id` is NOT an ever increasing number, but rather a
    counter - an integer relative to the image the segment belongs to - which
    resets with each newly captured image.

    The `segment_id` is a positive integer between 1 and 12, since each image
    has a total of 12 segments.

    """
    path = f'batch-{batch_id}-image-{image_id}-segment-{segment_id}.{ext}'
    assert 1 <= segment_id <= 12
    return os.path.join(cfg.SEGMENTS_DIR, path)


def store_captured_image(batch_id, image_id, image_obj):
    """Store the given `image_obj` on disk.

    The path under which the image is persisted on disk is deterministically
    decided upon given its corresponding `batch_id` and `image_id`, which is
    always a positive integer and resets per new batch.

    :param batch_id: the batch the image belongs to
    :param image_id: the image's position in the batch
    :param image_obj: the actual image - possibly a NumPy array

    :return: the path to the image on the filesystem

    """
    path = get_path_to_captured_image(batch_id, image_id)
    arr_write_safe(path, image_obj)
    return path


def store_annotated_image(batch_id, image_id, image_obj):
    """Store the processed/annotated `image_obj` on disk.

    This method behaves similarly to `store_captured_image` except
    that the file's path is suffixed differently.

    """
    path = get_path_to_annotated_image(batch_id, image_id)
    arr_write_safe(path, image_obj)
    return path


def store_extracted_segment(batch_id, image_id, segment_id, image_obj):
    """Store a segment extracted from an image on disk.

    :param batch_id: the batch the segment's image is belonging to
    :param image_id: the image this segment was extracted from
    :param segment_id: an integer denoting the segment - between 1 & 12
    :param image_obj: the actual image/segment - possibly a NumPy array

    :return: the path to the image on the filesystem

    """
    path = get_path_to_extracted_segment(batch_id, image_id, segment_id)
    arr_write_safe(path, image_obj)
    return path


def convert_template_to_img(npz_path, name="Template"):
    """!@brief Converts the template used for template matching to OpenCV image
    and shows it on screen.

    This function is needed for debugging purposes of the template matching
    step during segmentation. It shows the template on screen.

    @param npz_path: Path to .npz file of template
    @param name: Name of the window that the template will be displayed in
    """
    img = np.load(npz_path)
    cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    cv.imshow(name, img)


def save_mat_as_npz(src, path, compress=False):
    if compress:
        np.savez_compressed(path, a=src)
    else:
        np.savez(path, a=src)

    # try:
    #     test = np.load(path)
    #     assert np.array_equal(src, test['a'])
    # except AssertionError:
    #     log.error("Saving was not successful, please try again.")
    #     exit()

    log.info("Saving was successful! You can use the new template now.")


def save_mat(src, path):
    cv.imwrite(path, src)


def resize_tm(image, width=None, height=None, inter=cv.INTER_AREA):
    """
    Resize for template matching. Resizes an image according to the input
    width.
    Taken from https://github.com/agiledots/multiscale-template-matching

    :param image: the input image, an OpenCV mat containing the image data
    :param width: if resize based on width set ratio here
    :param height: if resize based on height set ratio here
    :param inter: the method to do interpolation
    :return resized: the resized input image
    """
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def get_path_to_heatmap(production_id, suffix='', ext='png'):
    """Get path to a heat map

    Returns a path to a heat map based on production id and program id (number
    of diamonds).

    :param production_id: The production id coming from OPC-UA server
    :return: save path for the currently computed heat map

    """
    path = f'production-{production_id}{suffix}.{ext}'
    return os.path.join(cfg.HEATMAPS_DIR, path)


def get_path_to_heatmap_inner_layers(production_id, ext='png'):
    """Get path to heat map

    Same as get_path_to_heatmap except that it returns the path for the inner
    layer.

    :param production_id: The production id coming from OPC-UA server
    :return: save path for the currently computed heat map
    """
    return get_path_to_heatmap(production_id, suffix='-inner', ext=ext)


def get_path_to_heatmap_outer_layers(production_id, ext='png'):
    """Get path to heat map

    Same as get_path_to_heatmap except that it returns the path for the outer
    layer.

    :param production_id: The production id coming from OPC-UA server
    :return: save path for the currently computed heat map
    """
    return get_path_to_heatmap(production_id, suffix='-outer', ext=ext)


def scale_segment(segment, ratio=cfg.DOWNSAMPLE_RATIO):
    """Scales a single segment according to the downsample ratio

    Scales the position of the bounding box and diamond positions such
    that it can be scaled in the frontend.

    :param segment: a single Segment instance
    :param ratio: the ratio you want to scale the bounding box and diamond
    positions by
    :return: scaled segment
    """
    segment.x = int(segment.x * ratio)
    segment.y = int(segment.y * ratio)
    segment.w = int(segment.w * ratio)
    segment.h = int(segment.h * ratio)
    for dia in segment.diamonds:
        dia.x = int(dia.x * ratio)
        dia.y = int(dia.y * ratio)

    return segment


def scale_segments(segments, ratio=cfg.DOWNSAMPLE_RATIO):
    """Scale method for list of segment instances

    :param segments: list of segment instances
    :param ratio: The downsample ratio used in the frontend
    :return: Returns the list of segments with updated bounding box and diamond
    positions
    """
    return [scale_segment(s, ratio=ratio) for s in segments]


def scale_image(img_array, height, width, save_to=None):
    """Scale the given image to the specified dimensions.

    The input image is scaled to (height, width, 3). If `save_to` is given,
    the result is persisted under it.

    """
    assert img_array.ndim == 3 and img_array.shape[-1] == 3
    img = Image.fromarray(img_array.astype(np.uint8))
    img = img.resize((width, height), resample=Image.BILINEAR)
    img_array_resized = np.asarray(img)
    if save_to:
        arr_write_safe(save_to, img_array_resized)
    return img_array_resized


def scale_image_by_ratio(img_array, ratio=cfg.DOWNSAMPLE_RATIO, save_to=None):
    """Scale the given image by the specified ratio"""
    height, width = img_array.shape[:2]
    assert 0 < ratio < 1
    assert not (height % (1 / ratio) or width % (1 / ratio))
    return scale_image(img_array, int(height * ratio), int(width * ratio),
                       save_to=save_to)


class Bin:
    def __init__(self, pt1, pt2):
        self.pt1 = pt1
        self.pt2 = pt2


class HeatMapRectangle:
    def __init__(self, id, pt1, pt2):
        self.id = id
        self.pt1 = pt1
        self.pt2 = pt2
        self.bins = []
        self.divide_into_bins()

    def divide_into_bins(self):
        y_old = self.pt1[1]
        i = 0
        for y in np.linspace(self.pt1[1], self.pt2[1], 4):
            j = 0
            x_old = self.pt1[0]
            for x in np.linspace(self.pt1[0], self.pt2[0], 4):
                if i != 0 and j != 0:
                    self.bins.append(Bin((int(x_old), int(y_old)),
                                         (int(x), int(y))))
                x_old = x
                j += 1
            y_old = y
            i += 1

    def color_bins(self):
        pass


def get_rectangles_for_heatmap(width, height):
    rectangles = []
    y_old = 0
    id = 1
    for i in np.linspace(1, 4, 4):
        x_old = 0
        y = int(height * (i / 4))
        for j in np.linspace(1, 3, 3):
            x = int(width * (j / 3))
            pt_1 = (int(x_old + (x - x_old) * 0.15),
                    int(y_old + (y - y_old) * 0.25))
            pt_2 = (int(x_old + (x - x_old) * 0.85),
                    int(y_old + (y - y_old) * 0.75))
            rectangles.append(HeatMapRectangle(id, pt_1, pt_2))
            x_old = x
            id += 1
        y_old = y

    return rectangles


def compute_diamond_bins(segment, verbose=False):
    """
    Dynamically computes which bin a detected diamond belongs to based on a
    segments position and size according to the graphic below.

    (x,y)       x1      x2      x3
        -------------------------
        |   1   |   2   |   3   |
    y1  -------------------------
        |   4   |   5   |   6   |
    y2  -------------------------
        |   7   |   8   |   9   |
    y3  -------------------------

    :param segment: A single segment instance
    :param verbose: Show diamond bins on/off for debugging
    :return: No return
    """
    # 1. divide segment in 9 bins
    x1 = int(segment.x + 0.36 * segment.w)
    x2 = int(segment.x + 0.65 * segment.w)
    x3 = int(segment.x + segment.w)

    y1 = int(segment.y + 0.38 * segment.h)
    y2 = int(segment.y + 0.58 * segment.h)
    y3 = int(segment.y + segment.h)
    segment.diamonds.sort(key=lambda dia: dia.y)

    if verbose:
        rgb_img = np.zeros((3000, 4096, 3), dtype=np.uint8)

    for dia in segment.diamonds:
        # bin 1
        if dia.x < x1 and dia.y < y1:
            dia.set_bin_id(1)
            if verbose:
                draw_circle(rgb_img, dia.x, dia.y, 10, consts.RED)
        # bin 2
        elif dia.x < x2 and dia.y < y1:
            dia.set_bin_id(2)
            if verbose:
                draw_circle(rgb_img, dia.x, dia.y, 10, consts.BLUE)
        # bin 3
        elif dia.x < x3 and dia.y < y1:
            dia.set_bin_id(3)
            if verbose:
                draw_circle(rgb_img, dia.x, dia.y, 10, consts.GREEN)
        # bin 4
        elif dia.x < x1 and dia.y < y2:
            dia.set_bin_id(4)
            if verbose:
                draw_circle(rgb_img, dia.x, dia.y, 10, consts.TEAL)
        # bin 5
        elif dia.x < x2 and dia.y < y2:
            dia.set_bin_id(5)
            if verbose:
                draw_circle(rgb_img, dia.x, dia.y, 10, consts.YELLOW)
        # bin 6
        elif dia.x < x3 and dia.y < y2:
            dia.set_bin_id(6)
            if verbose:
                draw_circle(rgb_img, dia.x, dia.y, 10, consts.PURPLE)
        # bin 7
        elif dia.x < x1 and dia.y < y3:
            dia.set_bin_id(7)
            if verbose:
                draw_circle(rgb_img, dia.x, dia.y, 10, consts.GREEN)
        # bin 8
        elif dia.x < x2 and dia.y < y3:
            dia.set_bin_id(8)
            if verbose:
                draw_circle(rgb_img, dia.x, dia.y, 10, consts.BLUE)
        # bin 9
        elif dia.x < x3 and dia.y < y3:
            dia.set_bin_id(9)
            if verbose:
                draw_circle(rgb_img, dia.x, dia.y, 10, consts.RED)

    if verbose:
        cv.imshow("bins", rgb_img)
        # cv.waitKey()


def write_heatmap(path, segments, width=int(4096 * cfg.DOWNSAMPLE_RATIO),
                  height=int(3000 * cfg.DOWNSAMPLE_RATIO), verbose=False):
    """Write heat map to disk

    Writes a heat map from a list of blob instances to disk. Also, if a
    previously saved heat map already exists it will be deleted since we want
    to update it.

    :param path: unique path of current heat map
    :param diamonds: a list of blob instances which hold x, y of a diamond
    position and a segment id and a bin id
    :param width: original width of input image multiplied by the
    DOWNSAMPLE_RATIO
    :param height: original height of input image multiplied by the
    DOWNSAMPLE_RATIO
    :param verbose: True if you want to show the heat map on screen using
    OpenCV
    :return: No Return
    """
    mkdir_p(cfg.HEATMAPS_DIR)
    if os.path.isfile(path):
        os.remove(path)

    # assert length % 12 and that all instance of Segment class
    assert len(segments) % 12 == 0
    assert all(isinstance(seg, Segment) for seg in segments)

    # create empty image
    heatmap_img = np.full((height, width, 3), 80, dtype=np.uint8)

    last_occurrences = [0] * 9 * 12
    length_to_split = [9] * 12

    last_occurrences = [list(islice(last_occurrences, length)) for length in
                        length_to_split]

    for i in range(len(segments) // 12):
        segment_list = segments[i * 12:(i + 1) * 12]

        # get 12 rectangles on empty image
        hm_rectangles = get_rectangles_for_heatmap(width, height)
        for seg_id, (rect, seg) in enumerate(zip(hm_rectangles, segment_list)):

            # initially create bounding box and set all diamond bins to red
            if i == 0:
                cv.rectangle(heatmap_img, rect.pt1, rect.pt2, (70, 70, 70), 4)
                for hm_bin in rect.bins:
                    cv.rectangle(heatmap_img, hm_bin.pt1, hm_bin.pt2,
                                 consts.RED, cv.FILLED)

            # fail safe: if no diamonds in a segment were detected it will be
            # skipped
            if not seg.diamonds:
                continue

            # compute diamonds bins
            compute_diamond_bins(seg, verbose=False)

            bin_occurrences = [dia.bin_id for dia in seg.diamonds]
            counted_occurrences = Counter(bin_occurrences)

            counted_occurrences = sorted(counted_occurrences.items(),
                                         key=lambda k: k[0])
            new_occurrences = [item[1] for item in counted_occurrences]
            last_occurrences[seg_id] = [sum(x)
                                        for x in zip(last_occurrences[seg_id],
                                                     new_occurrences)]
            max_occurrences = max(last_occurrences[seg_id])

            for hm_bin, bin_occurs in zip(rect.bins, last_occurrences[seg_id]):
                percentage = float(bin_occurs / max_occurrences)
                bin_color = (0,
                             int(255 * percentage),
                             int(255 * (1 - percentage)))
                cv.rectangle(heatmap_img, hm_bin.pt1, hm_bin.pt2, bin_color,
                             cv.FILLED)

        if verbose:
            name = "heat map"
            cv.namedWindow(name, cv.WINDOW_NORMAL)
            cv.resizeWindow(name, width, height)
            cv.imshow(name, heatmap_img)

    # save image and log path
    save_mat(heatmap_img, path)
    log.info('Generated and persisted heatmap: path=%s', path)


def show_img(name, src, divisor=4):
    """Displays an image for debugging

    :param name: Window Name
    :param src: cv.Mat (numpy array) containing the source data
    :param divisor: Divisor for the window size shown on screen.
    :return: No return
    """
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    height, width = src.shape[0] // divisor, src.shape[1] // divisor
    cv.resizeWindow(name, width, height)
    cv.imshow(name, src)


def draw_diamonds(img, segments):
    for seg in segments:
        for dia in seg.diamonds:
            draw_circle(img, dia.x, dia.y,
                        thickness=5, color=(0, 255, 0))


def plot_segments(img1, seg1, img2, seg2, img3, seg3, index):
    fig = plt.figure()
    fig.add_subplot(3, 1, 1)
    plt.imshow(img1[seg1.y:seg1.y + seg1.h, seg1.x:seg1.x + seg1.w])
    plt.title("Seg. %s - current status" % str(index + 1))
    fig.add_subplot(3, 1, 2)
    plt.imshow(img2[seg2.y:seg2.y + seg2.h, seg2.x:seg2.x + seg2.w])
    plt.title("Seg. %s - fixed w/ DBSCAN" % str(index + 1))
    fig.add_subplot(3, 1, 3)
    plt.imshow(img3[seg3.y:seg3.y + seg3.h, seg3.x:seg3.x + seg3.w])
    plt.title("Seg. %s - fixed w/o DBSCAN" % str(index + 1))
    fig.tight_layout()
