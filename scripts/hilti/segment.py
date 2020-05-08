class Segment:
    """Segment class

    A list of these objects is returned by the cv-pipeline. It contains all
    needed data to show the results of cv-pipeline in frontend.

    """
    def __init__(self, x, y, w, h):
        """Constructor of Segment class

        :param x: x coordinate of bounding box top left corner
        :param y: y coordinate of bounding box top right corner
        :param w: width of bounding box
        :param h: height of bounding box
        :return: No return
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.id = 0
        self.diamonds = []
        self.accuracy = 0.  # template matching accuracy
        self.pts = None
        self.left_border = 0
        self.right_border = 0
        self.bottom_border = 0

    def set_id(self, curr_id):
        """Set ID method
        Sets the ID according to position relative position of the segments in
        the image. ID is unique among one processed image.

        :param curr_id: The ID you want to give the specific segment object.
        """
        self.id = curr_id

    def get_nb_diamonds(self):
        """Returns amount of diamonds detected in this segment

        :return: Number of diamonds in this segment
        """
        return len(self.diamonds)

    def adjust_right_border(self, margin_r):
        self.right_border = margin_r

    def adjust_left_border(self, margin_l):
        self.left_border = margin_l

    def adjust_bottom_border(self, margin_b):
        self.bottom_border = margin_b
        
    def set_template_matching_accuracy(self, accuracy):
        """Set template matching accuracy
        Sets the template matching accuracy the segment was found with

        :param accuracy: The accuracy return by multi-scale template matching
        in cv-pipeline
        """
        self.accuracy = accuracy

    def __str__(self):
        """Overwrites __str__ method for debugging purposes.

        :return: Formatted print of this class
        """
        return ("Segment %i: [x=%i, y=%i, w=%i, h=%i], %i Diamonds, "
                "TM accuracy = %.2f%%" %
                (self.id, self.x, self.y, self.w, self.h,
                 self.get_nb_diamonds(), self.accuracy * 100))

    def set_pts(self, pts):
        self.pts = pts
