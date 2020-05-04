import cv2 as cv
import logging
import datetime as dt

from hilti.image2 import Image
from hilti.camera import get_next_image

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":

    # user-defined variables
    show_io = False
    save_io = True
    use_camera = False
    use_polygons = True
    use_border = False
    file_path = "/Users/mathian/Desktop/cv-pipeline-fixed_bounding_boxes/media/captures"

    # initialize Image class
    if use_border:
        border = 10
    else:
        border = 0
    curr_image = Image(use_polygons=use_polygons, border=border)

    nb = 0
    for i, j in enumerate(get_next_image(use_camera=use_camera,
                                         files=file_path)):
        file_path, img = j
        log.debug("This is image number %i.", i + 1)

        log.info(file_path)

        # cv-pipeline
        start = dt.datetime.now()
        segments = curr_image.run_pipeline(file_path, show_io=show_io, 
                                           create_rgb_img=True, save_io=save_io,)#
        end = dt.datetime.now()

        # Print results and timings
        for seg in segments:
            log.debug(seg)

        pipeline_time = (end - start).total_seconds()
        log.info(" Pipeline took %.2f seconds" % pipeline_time)

        if show_io:
            key = cv.waitKey()
            if "c" == chr(key & 255):
                log.info(" Next image")
            if "q" == chr(key & 255):
                break

    cv.destroyAllWindows()
    log.info(" Exiting")
