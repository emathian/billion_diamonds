import csv
import logging

log = logging.getLogger(__name__)


class CSVParser:
    def __init__(self):
        self.look_up_table = {"2156473": 940, "2156474": 940, "2156475": 940,
                              "2156476": 940, "2156477": 540, "2156478": 540,
                              "2156479": 394, "2156620": 441, "2145118": 544,
                              "2145119": 544, "2145580": 544, "2145581": 544,
                              "2145582": 472, "2145583": 472, "2145584": 394,
                              "2145585": 520}
        self.new_look_up_table = {"1": 68, "2": 123, "3": 63, "4": 134,
                                  "5": 109, "6": 107, "7": 45, "8": 74,
                                  "9": 51, "14": 63}
        self.csv_data = []

    def init_csv(self, folder_path):
        csv_path = folder_path + "metadata.csv"
        try:
            with open(csv_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                self.csv_data = list(csv_reader)
                # log.debug(self.csv_data)
        except FileNotFoundError:
            csv_path = folder_path + "../metadata.csv"
            with open(csv_path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=",")
                self.csv_data = list(csv_reader)

    def get_nbr_diamonds(self, img_path):
        isolated_path = img_path.split("/")[-1]
        split_path = isolated_path.split("_")
        file_name = split_path[1] + "_" + split_path[2]
        log.debug("file_name = %s" % file_name)
        nbr_diamonds = 0
        for sublist in self.csv_data:
            if sublist:
                if sublist[0] in file_name:
                    # nbr_diamonds = self.look_up_table.get(sublist[1], 0)
                    nbr_diamonds = self.new_look_up_table.get(sublist[4], 0)

        return nbr_diamonds

    def parse(self):
        pass
