import numpy as np
from glob import glob
from gvar import mean
from os import path


def guess_rod_orientation(folder_path):
    # Tale file BO čuden - mora delovati brez problemov krožnih importov pač
    # ZATO ŠELE TU IMPORTAŠ OSTALE DELE!!!
    from ..analysis.freq_and_phase_extract import freq_phase_ampl
    from ..analysis.import_trackdata import import_filepaths

    measurements = import_filepaths(list(glob(path.join(folder_path, "*.dat"))), guess_rod_orient=False)

    rod_orient = 0.
    phase_params = {'phase_start': 0, 'phase_end': 2 * np.pi}
    for meas in measurements:
        res = freq_phase_ampl(meas, fitting_params=phase_params)
        rod_orient += np.sin(-mean(res.rod_phase))
    return np.int0(np.sign(rod_orient))
