class Blob:
    """
    Blob class contains x and y coordinates of diamonds
    """
    def __init__(self, x, y, seg_id):
        """
        Initializer of Blob class.

        :param x: [int] x coordinate of diamond
        :param y: [int] y coordinate of diamond
        """
        self.x = x
        self.y = y
        self.seg_id = 0
        self.bin_id = 0

    def __str__(self):
        return "Blob: x = %i, y = %i" % (self.x, self.y)

    def __repr__(self):
        return "Blob: x = %i, y = %i" % (self.x, self.y)

    def __eq__(self, other):
        """For testing add method that overwrites __eq__"""
        return self.x == other.x and self.y == other.y

    def set_segment_id(self, seg_id):
        self.seg_id = seg_id

    def set_bin_id(self, bin_id):
        self.bin_id = bin_id
