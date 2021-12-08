# from cs506 import read
import pandas as pd


def read_csv(csv_file_path):
    """
    Given a path to a csv file, return a matrix (list of lists)
    in row major.
    """
    # read.read_csv(csv_file_path)

    return pd.read_csv(csv_file_path, header=None).values.tolist()
