import os
import tkfilebrowser
import re
from .MeasurementClass import Measurement
from ..utils.config import get_config


def select_data_dirs():
    initialdir = get_config()["meas"]
    return list(tkfilebrowser.askopendirnames(initialdir=initialdir))


def get_dotdat_filepaths_from_selected_dirs(selected_directories):
    filepaths = []

    for selected_dir in selected_directories:
        for dirpath, dirnames, filenames in os.walk(selected_dir, onerror=None):        # "." for current dir
            for filename in [f for f in filenames if f.endswith(".dat")]:
                filepaths.append(os.path.join(dirpath, filename))
    
    return filepaths


def filter_filepaths_by_ampl_offs_freq(filepaths, Iampl="*", Ioffs="*", Ifreq="*"):
    """" 
        This function filters the list of filepaths by Iampl, Ioffs and Ifreq.
        
        Input:
            filepaths   -->     List of absolute filepaths to data files

            Iampl       -->     The desired current amplitude (in mA). If Iampl = "*". The filter will consider all values of Iampl
            Ioffs       -->     The desired offset (in mA).            If Ioffs = "*". The filter will consider all values of Ioffs
            Ifreq       -->     The desired frequency (in mHz).        If Ifreq = "*". The filter will consider all values of Ifreq
        
        Returns:
            Filepaths according to the chosen input values.
    """

    if Ioffs == "*":
        Ioffs = r"\d+"
    if Iampl == "*":
        Iampl = r"\d+"
    if Ifreq == "*":
        Ifreq = r"\d+"

    pattern = re.compile("offs{0}_ampl{1}_freq{2}.dat".format(Ioffs, Iampl, Ifreq))

    filtered_filepaths = []
    for filepath in filepaths:
        filename = os.path.split(filepath)[1]       # Gets the tail of the filepath - aka. gets the filename
        if re.fullmatch(pattern, filename):
            filtered_filepaths.append(filepath)
    
    return filtered_filepaths


def filter_filepaths_by_keywords(filepaths, keyword_list=None):
    # Filters by keywords. The dirname must containt at least one of them to be selected.
    # If keyword_list is an empyt list, no filering is done!
    if keyword_list is None:
        return filepaths
    else:
        filtered_filepaths = []
        for filepath in filepaths:
            # Gets the tail of the head of the filepath - aka. gets the dirname in which the file is located
            dirname = os.path.split(os.path.split(filepath)[0])[1]
            for keyword in keyword_list:
                if keyword in dirname:
                    filtered_filepaths.append(filepath)
                    break
        
        return filtered_filepaths


def sort_filepaths_by_frequency(filepaths):
    pattern = re.compile(r"freq\d+.dat")
    return sorted(filepaths, key=lambda fp: int(re.search(pattern, fp).group()[4:-4]))


def import_filepaths(filepaths, **kwargs):
    measurements = []
    filepaths = sort_filepaths_by_frequency(filepaths)
    for filepath in filepaths:
        measurements.append(Measurement(filepath, **kwargs))
    # TODO: GROUP INTO MEASUREMENT RUNS 
    # TODO: Introduce the possibility of changing to different micorscope lenses --> differing pixelSize
    # FOR NOW - pixel_size hardcoded into the MeasurementObject
    return measurements


def select_filter_import_data(**kwargs):
    """
        This function lets you first select the data folders from which you want to import trackdata measurements,
        then it filters all found .dat filepaths by the current amplitude, offset and frequency. At last if
        filters the filepaths by keywoards. If a user wants to specify a keywoard that has to appear in all parent
        directory names of the .dat files, the keywoard/s should be put as strings into the keywoard list.

        The last step is to import all the filtered filepaths as a Measurement object, which holds information
        on the performed measurement.
    
        Returns the list of all selected measurement objects.
    """

    # Select data direcotries
    sdirs = select_data_dirs()

    # Filter filepaths
    filepaths = get_dotdat_filepaths_from_selected_dirs(sdirs)
    filepaths = filter_filepaths_by_ampl_offs_freq(filepaths, **{k: v for (k, v) in kwargs.items()
                                                                 if k in ["Ifreq", "Ioffs", "Iampl"]})
    filepaths = filter_filepaths_by_keywords(filepaths, kwargs.get("keyword_list", None))

    # Import data files as measurement objects
    measurements = import_filepaths(filepaths, **{k: v for (k, v) in kwargs.items()
                                                  if k in ["filter_duplicates", "max_num_points"]})

    return measurements
