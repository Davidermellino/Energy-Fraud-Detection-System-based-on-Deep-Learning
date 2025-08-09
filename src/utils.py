import csv
import numpy as np
from tqdm import tqdm


def read_csv_file(path):
    lines = []
    with open(path, "r", encoding="utf-16") as f:
        csvFile = csv.reader(f, delimiter="\t")

        i = 0

        for line in tqdm(csvFile, desc="Reading CSV file"):
            if i == 0:
                header = line
                i += 1
                continue
            lines.append(line)

    return header, np.array(lines)
