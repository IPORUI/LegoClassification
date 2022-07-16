import csv
import bpy
from generate_dataset import get_new_part, clear_parts

ROOT = r"C:\Users\sokfi\Code\jupyter\LegoClassification\\"
LDRAW_PATH = ROOT + r"dataset\ldraw\\"
OUT_PATH = ROOT + r"data\renders_2\\"
OUT_CSV = ROOT + r"data\RenderedParts_2.csv"
part_csv=ROOT + r'dataset\MostCommon.csv'

header = None
rows = []


def get_part_dims(part):
    return (1,1,1)


if __name__ == '__main__':
    with open(part_csv, 'r') as file:
        reader = csv.reader(file)

        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            rows.append(row)

            clear_parts()
            row.append(get_part_dims(get_new_part(name=row[2] + '.dat')))

    with open(part_csv, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        header.append('Dimensions')
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)