import cv2 as cv
import hilti.config as cfg


if __name__ == "__main__":
    window_name = "Template for Template Matching"
    template = cv.imread(cfg.template_path, cv.IMREAD_COLOR)

    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    cv.imshow(window_name, template)
    print("Showing template. Press any key to quit.")

    cv.waitKey()
    cv.destroyAllWindows()
