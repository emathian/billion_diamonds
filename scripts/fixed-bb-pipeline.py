import cv2 as cv
import logging
import datetime as dt
import numpy as np
from hilti.image2 import Image
from hilti.image2_ori import Image_ori
from hilti.camera import get_next_image
from hilti.utilities import show_img

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":

    # user-defined variables
    show_io = False
    save_io = True
    use_camera = False
    use_polygons = True
    use_border = False
    file_path = "/Users/mathian/Desktop/cv-pipeline-fixed_bounding_boxes/media/orig_dataset"

    # initialize Image class
    if use_border:
        border = 10
    else:
        border = 0
    curr_image = Image(use_polygons=True, border=border)
    curr_image_ori = Image_ori(use_polygons=False, border=border)

    nb = 0
    for i, j in enumerate(get_next_image(use_camera=use_camera,
                                         files=file_path)):
        file_path, img = j
        log.debug("This is image number %i.", i + 1)

        log.info(file_path)

        # cv-pipeline
        start = dt.datetime.now()
        segments, imgf = curr_image.run_pipeline(file_path, show_io=show_io, 
                                           create_rgb_img=True, save_io=save_io,)#

        segments_ori, imgf_ori = curr_image_ori.run_pipeline(file_path, show_io=show_io, 
                                           create_rgb_img=True, save_io=save_io,)

        fusion_img = 0.5 * imgf +  0.5 * imgf_ori
        fusion_img = fusion_img.astype(np.uint8)
        print("fusion_img", np.amin(fusion_img) , np.amax(fusion_img))
        #show_img("Final Output", fusion_img, divisor=4)
        cv.imwrite("fusion_orig_dataset/"+file_path.split('/')[-1], fusion_img)
        #cv.waitKey()

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
