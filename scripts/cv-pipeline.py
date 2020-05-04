import cv2 as cv
import logging
import datetime as dt

from hilti.image import Image
from hilti.camera import PATH_TO_FILES, get_next_image
from hilti.utilities import write_heatmap
from scripts.evaluation import Evaluator
from scripts.csv_parser import CSVParser

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


if __name__ == "__main__":

    # display results on current display
    show_io = True
    evaluate = False
    use_camera = False
    create_heatmap = True
    csv_available = True

    curr_image = Image()
    csv_parser = CSVParser()
    if csv_available:
        csv_parser.init_csv(PATH_TO_FILES)

    if evaluate:
        evaluator = Evaluator()
        mean_hit_rate = 0
        mean_error_rate = 0

    nb = 0
    heatmap_diamonds = []
    heatmap_segments = []
    dia_percentages = []
    for i, j in enumerate(get_next_image(use_camera=use_camera)):
        file_path, img = j
        log.debug("This is image number %i.", i + 1)

        # for evaluation
        if "labeled" in file_path:
            continue
        else:
            log.info(file_path)

        if csv_available:
            nbr_diamonds = csv_parser.get_nbr_diamonds(file_path)
            log.debug("There should be %i * 12 = %i diamonds."
                      % (nbr_diamonds, nbr_diamonds * 12))
        # if nbr_diamonds == 0:
        #     continue

        # cv-pipeline
        start = dt.datetime.now()
        segments = curr_image.run_pipeline(file_path, show_io=show_io)
        end = dt.datetime.now()

        # evaluation
        if evaluate:
            evaluator.set_true_segments(file_path, segments, verbose=False)
            tf, er = evaluator.compare_segments(segments, nb, verbose=False)
            evaluator.reset_vars()
            nb += 1
            mean_hit_rate += tf
            mean_error_rate += er
        # Print results and timings
        for seg in segments:
            log.debug(seg)
            # for heat map tests
            heatmap_diamonds.append(seg.diamonds)
            heatmap_segments.append(seg)
        pipeline_time = (end - start).total_seconds()
        log.info(" Pipeline took %.2f seconds" % pipeline_time)

        write_heatmap("./test.png", heatmap_segments, verbose=show_io)

        if csv_available:
            total_dias = sum([len(seg.diamonds) for seg in
                              curr_image.segments])
            if nbr_diamonds != 0:
                percentage = total_dias / (nbr_diamonds * 12)
                log.info("%i/%i Diamonds found, that corresponds %.2f%%" %
                         (total_dias, nbr_diamonds * 12, percentage * 100))
                dia_percentages.append(percentage)

        if show_io:
            key = cv.waitKey()
            save_name = file_path.split('/')[-1]
            cv.imwrite("./results/" + save_name + '.jpg', curr_image.rgb_img)
            if "c" == chr(key & 255):
                log.info(" Next image")
            if "q" == chr(key & 255):
                break

    if evaluate and nb != 0:
        log.info(" mean hit ratio = %.1f" %
                 (mean_hit_rate / (12 * nb) * 100))
        log.info(" mean error rate = %.1f" %
                 (mean_error_rate / (12 * nb) * 100))

    if create_heatmap:
        write_heatmap("./test.png", heatmap_segments, verbose=show_io)
        cv.waitKey()

    if csv_available and nb != 0:
        log.info("The total accuracy based on the nominal amount of diamonds "
                 "is %.2f%%" %
                 ((sum(dia_percentages) / len(dia_percentages)) * 100))
        log.info("This the list of percentages:")
        log.info(dia_percentages)

    cv.destroyAllWindows()
    log.info(" Exiting")
