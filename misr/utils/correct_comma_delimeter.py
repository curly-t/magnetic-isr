import numpy as np
from ..analysis.import_trackdata import select_data_dirs, get_dotdat_filepaths_from_selected_dirs


def number_with_comma_to_float(column):
    return bytes(column.decode("utf-8").replace(",", "."), "utf-8")


def correct_comma_delimeter():
    dirs = select_data_dirs()
    filepaths = get_dotdat_filepaths_from_selected_dirs(dirs)
    for filepath in filepaths:
        try:
            trackData = np.loadtxt(filepath, converters={1: number_with_comma_to_float, 3:number_with_comma_to_float})
            np.savetxt(filepath, trackData)
        except:
            print("Failed!\nThis folder maybe doesn't contain any comma ridden files left :)")
    print("All done for this directory!")

