import csv
import os
import subprocess

from fmt.utils import FileFormatError


class CSVFormatError(FileFormatError):
    def __init__(self, format, message="Incorrect file format for datafile. Format should be '.csv'"):
        super().__init__(format, message)


class CSVOperator:
    def __init__(self, fpath: str) -> None:

        self.fpath = fpath
        # check file format
        self.__check_file_format()

        # Set dummy file-handle
        self.fh = open(self.fpath, mode='a')

        # Set dummy mode
        # can be either 'a'- append, 'r'- read
        self.mode = 'a'
        self.rowcount = 0

        # Stores all the data read from a file
        self.data = None

    def __check_file_format(self):
        _, ext = os.path.splitext(self.fpath)
        if ext != '.csv':
            raise CSVFormatError(ext)

    def __close_fh(self):
        if self.fh is not None:
            self.fh.close()
            self.fh = None
            self.mode = None

    def read(self, store=False):
        if self.mode in ['a', None]:
            self.__close_fh()
            self.mode = 'r'
            self.fh = open(self.fpath, mode=self.mode)

        lr = None
        data = list(csv.reader(self.fh))
        self.rowcount = len(data)

        if self.rowcount > 1:
            if store:
                self.data = data
            lr = data[-1]
        else:
            print(f"File at ({self.fpath}) is empty!")

        self.__close_fh()
        return lr

    def writerow(self, row):
        if self.mode in ['r', None]:
            # print("Changing mode to write")
            self.__close_fh()
            self.mode = 'a'
            self.fh = open(self.fpath, mode=self.mode)

        writer = csv.writer(self.fh)
        writer.writerow(row)

        self.__close_fh()


class PathPlanningCSVOperator(CSVOperator):
    def __init__(self, alg: str, phase: str, extras: str = '') -> None:
        self.alg = alg
        self.extras = extras

        cmd = ["find", os.path.expanduser(
            '~'), "-type", "d", "-name", "CSC2630-Project"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        repo_dir = proc.communicate()[0].strip().decode('ascii')
        super().__init__(
            f"{repo_dir}/src/{self.alg}/output/{phase}/{self.alg}_{phase}{self.extras}_results.csv")

        self.phase = phase

        self.dcols = [
            'Overall Test Number',
            'Algorithm',
            'Map Type',
            'Map Id',
            'Start Point',
            'Goal Point',
            'Test Number',
            'Iteration',
            'Timestep',
            'Num Collision Checks',
            'Batch Size',
            'Cumulative Num Sampled',
            'Current Path Cost',
            'Any Path Found',
        ]

        self.__write_header()

    def __del__(self):
        self._CSVOperator__close_fh()

    def __write_header(self):
        # Write header if target csv is empty
        if os.stat(self.fpath).st_size == 0:
            self.writerow(self.dcols)

    def clear(self):
        open(self.fpath, 'w').close()
        self.__write_header()
        self._CSVOperator__close_fh()
