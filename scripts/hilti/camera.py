import os
import glob
import time
import cv2 as cv
import logging

#from pypylon import pylon, genicam


log = logging.getLogger(__name__)

# Path to test dataset on the local filesystem.
PATH_TO_FILES = ("/Users/mathian/Desktop/cv-pipeline-fixed_bounding_boxes/media/captures")


class Camera:
    def __init__(self, exposure_time=95000., buffer_size=10):
        # init and open camera device
        try:
            self.camera = pylon.InstantCamera(
                pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.MaxNumBuffer = buffer_size
            self.camera.Open()
            log.debug(" device = %s" % self.camera)
        except genicam.GenericException as e:
            log.error(" %s" % e)
            exit()

        path_to_settings = os.getcwd() + "\\camera-module\\src\\hilti\\" \
                                         "camera_settings.pfs"
        pylon.FeaturePersistence.Load(path_to_settings,
                                      self.camera.GetNodeMap(), True)

        # set exposure time critical var for diamond detection
        # self.camera.ExposureTime.SetValue(exposure_time)
        log.debug("Exposure time is %.1f" %
                  self.camera.ExposureTime.GetValue())

        # # sets trigger --> currently machine has to be manually triggered
        # # with button
        # NOTE trigger is now set via camera_settings.pfs
        # self.camera.TriggerSource.SetValue("Line1")
        # self.camera.TriggerMode.SetValue("On")

        # allows looping over images
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        # timeout for grab images in milliseconds
        self.timeout = pylon.waitForever

        # Converter
        self.converter = pylon.ImageFormatConverter()
        # self.converter.Gamma = 1.5  # why not writable?
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # if you want to save new settings try this
        # pylon.FeaturePersistence.Save("./camera_settings.pfs",
        #                               self.camera.GetNodeMap())

    def get_img(self):
        while self.camera.IsGrabbing():
            grab_result = self.camera.RetrieveResult(
                self.timeout, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                image = self.converter.Convert(grab_result)
                img = image.GetArray()
                # log.debug(img)
                yield img
            grab_result.Release()
        self.camera.Close()

    def __repr__(self):
        return self.camera.GetDeviceInfo().GetModelName()

    def __str__(self):
        return self.camera.GetDeviceInfo().GetModelName()

    def close_camera(self):
        self.camera.Close()


def get_next_image(max_sleep=5, use_camera=False, files=None):
    """Yield image objects to be processed.

    If `use_camera` is set to True, then images are grabbed from camera
    via the PyPylon Framework. Otherwise, they're read from disk under
    the given path, denoted by `files`, or `PATH_TO_FILES` if a path is
    not explicitly set.

    """
    if use_camera:
        for img in Camera().get_img():
            yield None, img
    else:
        for fpath in glob.glob(os.path.join(files or PATH_TO_FILES, '*.png')):
            #print("\n \n files ", files, "\n \n PATH_TO_FILES ", PATH_TO_FILES, "\n \n fpath  ",  fpath)
            start = time.time()
            if fpath.find("full-scale")!= -1:
                yield fpath, cv.imread(fpath)
                time.sleep(min(time.time() - start, max_sleep) % max_sleep)
