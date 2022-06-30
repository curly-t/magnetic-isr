import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev
import warnings
import math as m
from gvar import gvar, sdev
from gvar import mean as gvalue

from .import_trackdata import select_filter_import_data
from .ResultsClass import SingleResult


def running_average(input_array, averaging_len):
    """
        This function calculates the running average, for arbitrary averageing_len, above the input_array.
        The i-th element in the output array is the average of elements with indexs: i, i+1, i+2, ..., i + averageing_len.
        The output array is of length = len(input_array) - averageing_len + 1, because no periodic boundary is imposed on the data!
    """
    return np.convolve(input_array, np.ones(averaging_len)/averaging_len, mode='valid')


def identify_noninterupted_valid_idxs(data, low_limit, high_limit):
    # If data <= low_limit OR data >= high_limit - it is discarded.
    # Whats left is represented as a single array of all valid indexes
    # Operates with pure float data!!!
    return np.flatnonzero((gvalue(data) > low_limit) * (gvalue(data) < high_limit))


def convert_valid_idxs_to_valid_ranges(valid_idxs):
    # Gets a single array of all valid indexes and returns a 2D array, of valid range starts and ends
    starts = valid_idxs[np.flatnonzero(valid_idxs - (np.roll(valid_idxs, 1) + 1))]
    stops = valid_idxs[np.flatnonzero(valid_idxs - (np.roll(valid_idxs, -1) - 1))] + 1

    return np.concatenate((starts.reshape((len(starts), 1)), stops.reshape((len(stops), 1))), axis=1)


def concat_valid_data_and_time(data, time, valid_ranges):
    valid_data = []
    valid_time = []
    for valid in valid_ranges:
        valid_data.append(data[valid[0]:valid[1]])
        valid_time.append(time[valid[0]:valid[1]])
    return np.concatenate(valid_data), np.concatenate(valid_time)


def sinusoid_w_drift(t, A, freq, phase, const, speed):
    """
        The function used for fitting measured rod position and brightness data!
    """
    return A * np.sin(2*np.pi*freq * t + phase) + const + speed * t


def fit_sinus_w_drift(gvar_fit_data, gvar_fit_time, central_freq, timeLen,
                      phase_start=0.0, phase_end=2*np.pi, phase_init_guess=None):
    min_data = np.min(gvar_fit_data)
    max_data = np.max(gvar_fit_data)

    low_bound_data = gvalue([0., central_freq * 0.9988, phase_start - 0.05 * np.pi,
                             min_data, -(max_data - min_data) / timeLen])
    high_bound_data = gvalue([2. * (max_data - min_data) / 2., central_freq * 1.0012, phase_end + 0.05 * np.pi,
                              max_data, (max_data - min_data) / timeLen])

    if phase_init_guess is None:
        phase_init_guess = (phase_start + phase_end)/2.

    init_guess = (gvalue(max_data - min_data)/2., gvalue(central_freq), phase_init_guess,
                  gvalue(max_data + min_data)/2, 0.)

    try:
        data_coefs, data_cov = curve_fit(sinusoid_w_drift, gvalue(gvar_fit_time), gvalue(gvar_fit_data), p0=init_guess,
                                         sigma=sdev(gvar_fit_data), absolute_sigma=True,
                                         bounds=(low_bound_data, high_bound_data))
    except RuntimeError:
        print("The least squares minimization fitting failed!")

    chisq = np.sum(np.square(((gvalue(gvar_fit_data) - sinusoid_w_drift(gvalue(gvar_fit_time), *data_coefs))
                              /sdev(gvar_fit_data))))

    return gvar(data_coefs, data_cov), chisq


def get_drift_on_valid_ranges(gvar_data, measrmnt, valid_ranges, data_coefs):
    averaging_len = int(measrmnt.numFrames / gvalue(data_coefs[1] * measrmnt.timeLength))
    avging = np.zeros(len(valid_ranges))
    drifts = []
    times = []
    new_valid_ranges = np.copy(valid_ranges)
    for i, valid in enumerate(valid_ranges):
        if 2 < averaging_len < 0.5 * (valid[1] - valid[0]):
            # OK avg_len
            avging[i] = 1
            drifts.append(running_average(gvar_data[valid[0]: valid[1]], averaging_len))
            drift_start_offset = averaging_len//2
            drift_stop_offset = -(averaging_len - averaging_len//2) + 1

            times.append(measrmnt.times[valid[0]: valid[1]][drift_start_offset:drift_stop_offset])
            new_valid_ranges[i] += np.array([drift_start_offset, drift_stop_offset - 1])
        else:
            # Too short or too long avg_len - one should use just the values given with fitting of sinus + simple drift
            drifts.append(data_coefs[3] + data_coefs[4] * measrmnt.times[valid[0]: valid[1]])
            times.append(measrmnt.times[valid[0]: valid[1]])

    return np.concatenate(drifts), np.concatenate(times), new_valid_ranges


def smooth_drift(drift, time, eval_times, smoothing_factor=0.1):
    # SAVGOL NE MORE FITTAT UNREGULAR DATAPOINTS - SPLINE PA SEVEDA LAHKO!!! :)
    spline_rep_tuple, fp, ier, msg = splrep(gvalue(time), gvalue(drift), w=1/sdev(drift), k=3, s=smoothing_factor * len(drift), full_output=True)
    return splev(gvalue(eval_times), spline_rep_tuple)


def determine_bandreject_params(data_coefs, expected_freq, factor):
    fit_freq = data_coefs[1]

    bandreject_center = gvalue(fit_freq)

    # freq_discrep = fit_freq - expected_freq
    # bandreject_width = factor * (np.abs(gvalue(freq_discrep)) + np.abs(2*sdev(freq_discrep)))
    bandreject_width = factor * bandreject_center

    return bandreject_width, bandreject_center


def remove_freqs_near_expected_freq(drift, time, bandreject_width, bandreject_center):
    if len(drift) % 2 != 0:
        # Če je dolžina liha: naredi jo sodo samo za tale del
        whole_drift = drift[:]
        drift = drift[:-1]
        time = time[:-1]
    else:
        whole_drift = drift

    coefs = np.polyfit(gvalue(time), gvalue(drift), deg=1)
    polyapprox = np.poly1d(coefs)
    fft_ampls = np.fft.rfft(gvalue(drift) - polyapprox(gvalue(time)))
    fft_freqs = np.fft.rfftfreq(len(drift), d=gvalue(time[1] - time[0]))

    win_center = np.argmin(np.abs(bandreject_center - fft_freqs))
    win_freq_start = bandreject_center - bandreject_width/2
    fft_freq_steps = fft_freqs[1] - fft_freqs[0]

    win_width = 2 * int((bandreject_center - win_freq_start) / fft_freq_steps) + 1

    win_ideal_start = int(win_center - (win_width - 1)/2)
    win_ideal_stop = int(win_center + (win_width - 1)/2 + 1)

    win_start = max(0, win_ideal_start)
    win_stop = min(len(fft_freqs), win_ideal_stop)

    edge_points = np.array([win_start, win_stop-1])
    linecoefs = np.polyfit(fft_freqs[edge_points], fft_ampls[edge_points], deg=1)
    line = np.poly1d(linecoefs)
    fft_ampls[win_start:win_stop] = line(fft_freqs[win_start:win_stop])

    if len(whole_drift) % 2 != 0:
        res = np.zeros_like(whole_drift)
        res[:-1] = np.fft.irfft(fft_ampls) + polyapprox(gvalue(time))
        res[-1] = whole_drift[-1]
        return res
    else:
        return np.fft.irfft(fft_ampls) + polyapprox(gvalue(time))


def subtract_drift_from_data_on_valid_ranges(drift, data, time, valid_ranges):
    return concat_valid_data_and_time(data - drift, time, valid_ranges)


def perform_fitting_on_data(gvar_data, measrmnt, expected_freq, low_limit, high_limit,
                            smoothing_factor=10, bandreject_width_factor=0.5, plot_results=False,
                            phase_start=0.0, phase_end=2*np.pi, bandfilter_drift=True, num_final_fits=20):
    if plot_results:
        fig, (ax_data, ax_drift) = plt.subplots(1, 2)
        ax_data.plot(gvalue(measrmnt.times), gvalue(gvar_data), 'y-', label="Data")
        plt.suptitle(f"freq={gvalue(measrmnt.Ifreq):.3f}")
    valid_indexes = identify_noninterupted_valid_idxs(gvar_data, low_limit, high_limit)
    valid_ranges = convert_valid_idxs_to_valid_ranges(valid_indexes)        # BUG - ERRORS OUT
    valid_data, valid_times = concat_valid_data_and_time(gvar_data, measrmnt.times, valid_ranges)
    if plot_results:
        ax_data.plot(gvalue(valid_times), gvalue(valid_data), 'g--', label="Valid data")
        ax_data.legend()
    # Določi frekvenco in const odmik + speed, zato točna faza še ni tulk pomembna, frekvenca ej itaka ful blizu Ifreq
    # domik pa je lahko določen tudi brez točne faze!
    data_coefs, _ = fit_sinus_w_drift(valid_data, valid_times, expected_freq, measrmnt.timeLength, phase_start, phase_end)
    valid_drift, new_valid_times, new_valid_ranges = get_drift_on_valid_ranges(gvar_data, measrmnt, valid_ranges, data_coefs)
    if plot_results:
        ax_drift.plot(gvalue(new_valid_times), gvalue(valid_drift), "b-", label="Valid drift")
    smoothed_drift = smooth_drift(valid_drift, new_valid_times, measrmnt.times, smoothing_factor)
    if plot_results:
        ax_drift.plot(gvalue(measrmnt.times), gvalue(smoothed_drift), 'y--', label="Smooth extended drift")
        if not bandfilter_drift:
            ax_drift.legend()
            plt.show()

    if bandfilter_drift:
        bandreject_width, bandreject_center = determine_bandreject_params(data_coefs, expected_freq, bandreject_width_factor)
        smoothed_drift = remove_freqs_near_expected_freq(smoothed_drift, measrmnt.times, bandreject_width,
                                                                bandreject_center)
        if plot_results:
            ax_drift.plot(gvalue(measrmnt.times), gvalue(smoothed_drift), "k--", label="Final drift")
            ax_drift.legend()
            plt.show()

    valid_driftless_data, valid_driftless_times = subtract_drift_from_data_on_valid_ranges(smoothed_drift,
                                                                                           gvar_data,
                                                                                           measrmnt.times,
                                                                                           new_valid_ranges)
    # Multiple fittings - to ensure best fit, use lowest chisq fit!
    finalfit_results = np.ones(num_final_fits) * np.inf
    all_data_coefs = []
    initial_phase_guesses = np.linspace(phase_start, phase_end, num=num_final_fits)
    for i, initial_phi in enumerate(initial_phase_guesses):
        data_coefs, chisq = fit_sinus_w_drift(valid_driftless_data, valid_driftless_times, expected_freq,
                                              measrmnt.timeLength, phase_start, phase_end, phase_init_guess=initial_phi)
        finalfit_results[i] = chisq
        all_data_coefs.append(data_coefs)

    data_coefs = all_data_coefs[np.argmin(finalfit_results)]

    if plot_results:
        plt.plot(gvalue(valid_driftless_times), gvalue(valid_driftless_data), 'b-', label="Driftless data")
        plt.plot(gvalue(valid_driftless_times), gvalue(sinusoid_w_drift(valid_driftless_times, *data_coefs)), 'k-', label="Fit")
        plt.legend()
        plt.suptitle(f"freq={gvalue(measrmnt.Ifreq):.3f}")
        plt.show()

    return data_coefs


def check_for_freq_discrepancy(bright_coefs, pos_coefs, func_params):
    # Check for frequency discrepancy
    relative_discrep = gvalue((bright_coefs[1] - pos_coefs[1])/bright_coefs[1])
    if m.fabs(relative_discrep) >= func_params["freq_err"]:
        warnings.warn(f"Rel. freq. err: Brightness freq: {bright_coefs[1]} Position freq: {pos_coefs[1]}")


def determine_phase_start_end(bright_phase, rod_orientation, fitting_params):
    if ("phase_start" in fitting_params) and ("phase_end" in fitting_params):
        ps, pe = fitting_params["phase_start"], fitting_params["phase_end"]
        exclude_phases = ["phase_start", "phase_end"]
        new_fit_params = {k: v for (k, v) in fitting_params.items() if k not in exclude_phases}
    else:
        # Phase sign is different for different rod orientations mu +- 1
        new_fit_params = fitting_params
        if rod_orientation == 1:
            ps, pe = gvalue(bright_phase), gvalue(bright_phase + np.pi)   # mu = +1 (apparent zaostajanje)
        else:
            ps, pe = gvalue(bright_phase) - np.pi, gvalue(bright_phase)   # mu = -1 (apparent PREHITEVANJE)
    return ps, pe, new_fit_params


def freq_phase_ampl(measrmnt, fpa_params={}, fitting_params={}):
    """ This function calculates the frequency of the vibration of the rod, the relative phase between
        the brightness modulation (current) and the response of the rod (position), and the amplitude of the rod response,
        from the results of rod tracking, and brightness logging.
        
        Inputs:
            measrmnt            --> The Measurement data object.
                measrmnt.trackData   ...    The data from rod tracking under the influence of changing magnetic fields.

            freq_err            --> The maximum relative discrepancy permitted between measured brihtness modulation and rod position modulation frequency.
                                    If the relative discrepancy is greater, the function returns a UserWarning

            plot_track_results  --> A boolean value used to determine, weather to plot the results from tracking rod's position
            
            plot_bright_results --> A boolean value used to determine, weather to plot the results from brightnes analisys

            bandfilter_smooth_drift  -->  If True, an aditional bandreject filter is applied on the spline
                                          repr. of drift. Best to leave False


        Returns:
            The function returns a SingleResult class object with the following fields:

            rod_freq        -->     The frequencie of the vibration
            rod_freq_err    -->     The absolute error in frequencies
            rod_phase       -->     The relative phase (in radians) between the position and curent modulation (brightness).
            rod_phase_err   -->     The error of relative phase
            rod_ampl        -->     The amplitude of the rod position modulation (in pixels)
            rod_ampl_err    -->     The absolute error in the amplitude of the rod position modulation

        """

    func_params = {"plot_bright_results": False, "plot_track_results": False,
                   "freq_err": 0.1}
    func_params.update(fpa_params)
    print("-"*20)
    print(f"Current freq {measrmnt.Ifreq}")
    if measrmnt.Ifreq == 0:
        rod_phase = gvar(5e-3, 5e-3)
        rod_freq = gvar(1e-4, 1e-4)
        rod_ampl = gvar(measrmnt.zero_ampl, 0.5)     # Pol pixla
        mean_rod_position = gvar(measrmnt.zero_mean, 0.5)   # Tudi pol pixla
    else:
        bright_coefs = perform_fitting_on_data(measrmnt.brights, measrmnt, measrmnt.Ifreq, 0, 255,
                                               plot_results=func_params["plot_bright_results"], **fitting_params)
        print(f"Brightness freq, phase {bright_coefs[1]} {bright_coefs[2]}")

        ps, pe, new_fit_params = determine_phase_start_end(bright_coefs[2], measrmnt.rod_orient, fitting_params)
        pos_coefs = perform_fitting_on_data(measrmnt.positions, measrmnt, bright_coefs[1], 1, measrmnt.x_pixels - 1,
                                            plot_results=func_params["plot_track_results"],
                                            phase_start=ps, phase_end=pe,
                                            **new_fit_params)
        print(f"Rod freq, phase {pos_coefs[1]} {pos_coefs[2]}")
        # ADD AN ADDITIONAL PHASE ERROR --> FREQUENCY AND PHASE OF THE BRIGHTNESS IMPACT THE PHASE GREATLY!!!!!!!!

        check_for_freq_discrepancy(bright_coefs, pos_coefs, func_params)
        mean_rod_position = np.sum(measrmnt.positions) / len(measrmnt.positions)

        if measrmnt.rod_orient == 1:
            rod_phase = pos_coefs[2] - bright_coefs[2] - np.pi
        else:
            rod_phase = pos_coefs[2] - bright_coefs[2]
        rod_freq = pos_coefs[1]
        rod_ampl = pos_coefs[0]

    result_dict = {"rod_freq": rod_freq, "rod_ampl": rod_ampl, "rod_phase": rod_phase}

    return SingleResult(result_dict, measrmnt, mean_rod_position)


def select_and_analyse(import_params={}, **kwargs):

    """Returns a list of all acceptable system responses in selected direcotries"""

    default_import_params = {"filter_wierd_phases": True}
    default_import_params.update(import_params)
    import_params = default_import_params

    measurements = select_filter_import_data(**{k: v for (k, v) in import_params.items() if k != "filter_wierd_phases"})
    system_responses = []
    for measrmnt in measurements:
        sys_resp = freq_phase_ampl(measrmnt,
                                   fpa_params=kwargs.get("fpa_params", {}),
                                   fitting_params=kwargs.get("fitting_params", {}))

        if import_params["filter_wierd_phases"]:
            if not sys_resp.phase_sanity_check():
                system_responses.append(sys_resp)
            else:
                print("Measurement discarded because of insane phase!")
        else:
            system_responses.append(sys_resp)

    return system_responses
