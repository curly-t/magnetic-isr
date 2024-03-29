import numpy as np
from gvar import gvar, mean
from .config import get_config
from ..analysis.freq_and_phase_extract import select_and_analyse, freq_phase_ampl
from ..analysis.import_trackdata import import_filepaths
from tkfilebrowser import askopenfilenames, askopendirname, askopendirnames
import os.path as opath
from os import walk
from glob import glob


def import_results(filepaths=None):
    """ Returns a dictionary of 2D numpy arrays (shown below), of all data selected in form of .txt files.
        {"label": np.array([[freq AR phase]1, [freq AR phase]2, ...])} """

    if filepaths is None:
        meas_folder = get_config()["meas"]
        filepaths = askopenfilenames(initialdir=meas_folder, title="Select one or more .txt files to import!")
    responses = {}
    for fp in filepaths:
        responses[opath.split(fp)[1]] = np.loadtxt(fp)
    return responses


def save_responses_to_txt(responses, filepath, mode="a"):
    with open(filepath, mode) as f:
        for point_res in responses:
            print(point_res.rod_freq.mean, point_res.AR.mean, point_res.rod_phase.mean, point_res.rod_freq.sdev, point_res.AR.sdev, point_res.rod_phase.sdev, file=f)


def reanalyze_responses():
    # Grozna imena spremenljiv, neintuitiven program, ampak za zdle bo molgo bit vredu!
    meas_folder = get_config()["meas"]
    export_folder = askopendirname(initialdir=meas_folder, title="Select export folder!")

    folders_for_reanalysis = askopendirnames(initialdir=meas_folder, title="Select folders to reanalyze!")
    all_subfolders = []
    for folder in folders_for_reanalysis:
        all_subfolders += [x[0] for x in walk(folder)]

    valid_datafolders = []
    for folder in all_subfolders:
        folder_files = [folder] + [glob(opath.join(folder, "*.dat"))]
        if len(folder_files[1]) > 1:
            valid_datafolders.append(folder_files)

    for valid_folder in valid_datafolders:
        experiment_foldername = opath.split(valid_folder[0])[1]
        date_foldername = opath.split(opath.split(valid_folder[0])[0])[1]
        full_export_filename = opath.join(export_folder, f"{date_foldername}_{experiment_foldername}.txt")

        measurements = import_filepaths(valid_folder[1])
        responses = []
        for meas in measurements:
            print(f"Working on: {meas.dirname}")
            responses.append(freq_phase_ampl(meas))
        responses = sorted(responses, key=lambda resp: resp.rod_freq.mean)  # Sortiranje po frekvencah
        save_responses_to_txt(responses, full_export_filename)
