import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import splrep, splev
import scipy.signal.windows as ssw
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


def convert_valid_idxs_to_valid_ranges(valid_idxs, data_len):
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


def fit_sinus_w_drift(gvar_fit_data, gvar_fit_time, central_freq, timeLen, phase_start=0.0, exceptable_phase_insanity=0.0):
    min_data = np.min(gvar_fit_data)
    max_data = np.max(gvar_fit_data)

    low_bound_data = gvalue([0.5 * (max_data - min_data) / 2., central_freq * 0.8,
                             phase_start,
                             min_data, -(max_data - min_data) / timeLen])
    high_bound_data = gvalue([1.2 * (max_data - min_data) / 2., central_freq * 1.2,
                              phase_start + 2 * np.pi + exceptable_phase_insanity,
                              max_data, (max_data - min_data) / timeLen])

    try:
        data_coefs, data_cov = curve_fit(sinusoid_w_drift, gvalue(gvar_fit_time), gvalue(gvar_fit_data),
                                             sigma=sdev(gvar_fit_data), absolute_sigma=True,
                                             bounds=(low_bound_data, high_bound_data))
    except RuntimeError:
        print("The least squares minimization fitting failed!")

    return gvar(data_coefs, data_cov)


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
            drifts.append(running_average(gvar_data[valid[0] : valid[1]], averaging_len))
            drift_start_offset = averaging_len//2
            drift_stop_offset = -(averaging_len - averaging_len//2) + 1

            times.append(measrmnt.times[valid[0] : valid[1]][drift_start_offset:drift_stop_offset])
            new_valid_ranges[i] += np.array([drift_start_offset, -drift_stop_offset])
        else:
            # Too short or too long avg_len - one should use just the values given with fitting of sinus + simple drift
            drifts.append(data_coefs[3] + data_coefs[4] * measrmnt.times[valid[0] : valid[1]])
            times.append(measrmnt.times[valid[0] : valid[1]])

    return np.concatenate(drifts), np.concatenate(times), new_valid_ranges


def smooth_drift(drift, time, eval_times, smoothing_factor=0.1):
    # SAVGOL NE MORE FITTAT UNREGULAR DATAPOINTS - SPLINE PA SEVEDA LAHKO!!! :)
    spline_rep_tuple, fp, ier, msg = splrep(gvalue(time), gvalue(drift), w=1/sdev(drift), k=3, s=smoothing_factor * len(drift), full_output=True)
    return splev(gvalue(eval_times), spline_rep_tuple)


def determine_bandreject_params(data_coefs, expected_freq, factor):
    fit_freq = data_coefs[1]

    bandreject_center = gvalue((fit_freq + expected_freq) / 2)

    freq_discrep = fit_freq - expected_freq
    bandreject_width = factor * (np.abs(gvalue(freq_discrep)) + np.abs(2*sdev(freq_discrep)))

    return bandreject_width, bandreject_center


def remove_freqs_near_expected_freq(drift, time, bandreject_width, bandreject_center):
    fft_ampls = np.fft.rfft(gvalue(drift))
    fft_freqs = np.fft.rfftfreq(len(drift), d=gvalue(time[1] - time[0]))

    win_freq_start = bandreject_center - bandreject_width/2
    win_freq_stop = bandreject_center + bandreject_width/2
    win_start = np.argmin(np.abs(fft_freqs - win_freq_start))
    win_stop = np.argmin(np.abs(fft_freqs - win_freq_stop))
    win_center = np.argmin(np.abs(fft_freqs - bandreject_center))

    fft_freq_steps = fft_freqs[1] - fft_freqs[0]

    win_width = int((win_freq_stop - win_freq_start) / fft_freq_steps) + 1
    window = 1 - ssw.kaiser(win_width, beta=7)

    win_left_cut = int(np.abs(win_freq_start - fft_freqs[win_start]) // fft_freq_steps)
    win_right_cut = win_width - int(np.abs(win_freq_stop - fft_freqs[win_stop]) // fft_freq_steps)

    bandreject_filter = np.ones(len(fft_freqs))
    bandreject_filter[win_start:win_stop + 1] = window[win_left_cut:win_right_cut]
    plt.plot(bandreject_filter, label="window")
    plt.legend()
    plt.show()

    filtered = np.fft.irfft(fft_ampls * bandreject_filter)
    plt.plot(filtered, label="filtered")
    plt.plot(drift, label="input")
    plt.legend()
    plt.show()
    return filtered


def subtract_drift_from_data_on_valid_ranges(drift, data, time, valid_ranges):
    return concat_valid_data_and_time(data - drift, time, valid_ranges)


def freq_phase_ampl(measrmnt, freq_err=0.1, plot_track_results=False,
    plot_bright_results=False, bandfilter_smooth_drift=False, rod_led_phase_correct=True, exceptable_phase_insanity=0.1*np.pi):
    """ This function calculates the frequency of the vibration of the rod, the relative phase between
        the brightness modulation (current) and the response of the rod (position), and the amplitude of the rod response,
        from the results of rod tracking, and brightness logging.
        
        Inputs:
            measrmnt            --> The Measurement data object.
                measrmnt.trackData   ...    The data from rod tracking under the influence of changing magnetic fields.

                                            The trackData array is comprised of 4 columns:
                                            frameIdx, frameTime, rodEdgePos, brightnes

                                            Time is logged in seconds, and rodEdgePos in pixels.
                                            Brightness has arbitrary linear values.

                                            ***************************************************************************
                                            ALWAYS USE measrmnt.positions (gvars)
                                            instead of measrnmnt.trackData[:, 2] (floats!!!)
                                            ***************************************************************************

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

    # FITTING of BRIGHTNESS DATA ---------------------------------------------------------------------------------------
    if plot_bright_results:
        fig, (ax_bright, ax_drift) = plt.subplots(1, 2)
        ax_bright.plot(gvalue(measrmnt.times), gvalue(measrmnt.brights), 'y-', label="Bright")
        plt.suptitle(f"Brightness, freq={gvalue(measrmnt.Ifreq):.3f}")
    valid_indexes = identify_noninterupted_valid_idxs(measrmnt.brights, 0, 255)
    valid_ranges = convert_valid_idxs_to_valid_ranges(valid_indexes, len(measrmnt.brights))
    valid_brights, valid_times = concat_valid_data_and_time(measrmnt.brights, measrmnt.times, valid_ranges)
    if plot_bright_results:
        ax_bright.plot(gvalue(valid_times), gvalue(valid_brights), 'g--', label="Valid")
        ax_bright.legend()
    bright_coefs = fit_sinus_w_drift(valid_brights, valid_times, measrmnt.Ifreq, measrmnt.timeLength)
    valid_drift, new_valid_times, new_valid_ranges = get_drift_on_valid_ranges(measrmnt.brights, measrmnt, valid_ranges, bright_coefs)
    if plot_bright_results:
        ax_drift.plot(gvalue(new_valid_times), gvalue(valid_drift), "b-", label="valid bright drift")
    smoothed_drift = smooth_drift(valid_drift, new_valid_times, measrmnt.times, 50.)
    if plot_bright_results:
        ax_drift.plot(gvalue(measrmnt.times), gvalue(smoothed_drift), 'y--', label="Smooth extended bright drift")
        if not bandfilter_smooth_drift:
            ax_drift.legend()
            plt.show()

    if bandfilter_smooth_drift:
        bandreject_width, bandreject_center = determine_bandreject_params(bright_coefs, measrmnt.Ifreq, 10)
        smoothed_drift = remove_freqs_near_expected_freq(smoothed_drift, measrmnt.times, bandreject_width,
                                                                bandreject_center)
        if plot_bright_results:
            ax_drift.plot(gvalue(measrmnt.times), gvalue(smoothed_drift), "k--", label="final drift")
            ax_drift.legend()
            plt.show()

    valid_driftless_brights, valid_driftless_times = subtract_drift_from_data_on_valid_ranges(smoothed_drift,
                                                                                              measrmnt.brights,
                                                                                              measrmnt.times,
                                                                                              new_valid_ranges)
    bright_coefs = fit_sinus_w_drift(valid_driftless_brights, valid_driftless_times, measrmnt.Ifreq, measrmnt.timeLength)
    if plot_bright_results:
        plt.plot(gvalue(valid_driftless_times), gvalue(valid_driftless_brights), 'b-', label="Driftless Brights")
        plt.plot(gvalue(valid_driftless_times), gvalue(sinusoid_w_drift(valid_driftless_times, *bright_coefs)), 'k-', label="Fit")
        plt.legend()
        plt.suptitle(f"Brightness, freq={gvalue(measrmnt.Ifreq):.3f}")
        plt.show()
    # END --------------------------------------------------------------------------------------------------------------



    # FITTING of POSITION DATA -----------------------------------------------------------------------------------------
    if plot_track_results:
        fig, (ax_position, ax_drift) = plt.subplots(1, 2)
        ax_position.plot(gvalue(measrmnt.times), gvalue(measrmnt.positions), 'y-', label="Positin")
        plt.suptitle(f"Positions, freq={gvalue(measrmnt.Ifreq):.3f}")
    valid_indexes = identify_noninterupted_valid_idxs(measrmnt.positions, 1, measrmnt.x_pixels-1)
    valid_ranges = convert_valid_idxs_to_valid_ranges(valid_indexes, len(measrmnt.positions))
    valid_positions, valid_times = concat_valid_data_and_time(measrmnt.positions, measrmnt.times, valid_ranges)
    if plot_track_results:
        ax_position.plot(gvalue(valid_times), gvalue(valid_positions), 'g--', label="Valid")
        ax_position.legend()
    pos_coefs = fit_sinus_w_drift(valid_positions, valid_times, measrmnt.Ifreq, measrmnt.timeLength)
    valid_drift, new_valid_times, new_valid_ranges = get_drift_on_valid_ranges(measrmnt.positions, measrmnt, valid_ranges,
                                                                               pos_coefs)
    if plot_track_results:
        ax_drift.plot(gvalue(new_valid_times), gvalue(valid_drift), "b-", label="valid pos drift")
    smoothed_drift = smooth_drift(valid_drift, new_valid_times, measrmnt.times, 50.)
    if plot_track_results:
        ax_drift.plot(gvalue(measrmnt.times), gvalue(smoothed_drift), 'y--', label="Smooth extended pos drift")
        if not bandfilter_smooth_drift:
            ax_drift.legend()
            plt.show()

    if bandfilter_smooth_drift:
        bandreject_width, bandreject_center = determine_bandreject_params(pos_coefs, measrmnt.Ifreq, 10)
        smoothed_drift = remove_freqs_near_expected_freq(smoothed_drift, measrmnt.times, bandreject_width,
                                                                bandreject_center)
        if plot_track_results:
            ax_drift.plot(gvalue(measrmnt.times), gvalue(smoothed_drift), "k--", label="final drift")
            ax_drift.legend()
            plt.show()

    valid_driftless_positions, valid_driftless_times = subtract_drift_from_data_on_valid_ranges(smoothed_drift,
                                                                                              measrmnt.positions,
                                                                                              measrmnt.times,
                                                                                              new_valid_ranges)
    pos_coefs = fit_sinus_w_drift(valid_driftless_positions, valid_driftless_times, measrmnt.Ifreq,
                                     measrmnt.timeLength)
    if plot_track_results:
        plt.plot(gvalue(valid_driftless_times), gvalue(valid_driftless_positions), 'b-', label="Driftless Position")
        plt.plot(gvalue(valid_driftless_times), gvalue(sinusoid_w_drift(valid_driftless_times, *pos_coefs)), 'k-',
                 label="Fit")
        plt.legend()
        plt.suptitle(f"Positions, freq={gvalue(measrmnt.Ifreq):.3f}")
        plt.show()
    # END --------------------------------------------------------------------------------------------------------------


    # Check for frequency discrepancy
    relative_discrep = gvalue((bright_coefs[1] - pos_coefs[1])/bright_coefs[1])
    if m.fabs(relative_discrep) >= freq_err:
        warnings.warn("Relativna razlika v frekvencah dobljenih iz brightness data in position data je vecja od {0}".format(freq_err) +
        "\nBrightness freq: {0}\nPosition freq: {1}".format(bright_coefs[1], pos_coefs[1]))

    result_dict = {"rod_freq": pos_coefs[1], "rod_ampl": pos_coefs[0], "rod_phase": pos_coefs[2] - bright_coefs[2]}

    # BECAUSE PHASE OF THE ROD AND PHASE OF THE LED ARE RECORDED DIFFERENTLY - THERE APPEARS AN ADDITIONAL 180 PHASE DIFFERENCE
    # CORRECT FOR PHASE DIFFERENCE
    if rod_led_phase_correct:
        result_dict["rod_phase"] += np.pi

    mean_rod_position = np.sum(measrmnt.positions)/len(measrmnt.positions)

    return SingleResult(result_dict, measrmnt, exceptable_phase_insanity, mean_rod_position)


def select_and_analyse(Iampl="*", Ioffs="*", Ifreq="*", keyword_list=[], bandfilter_smooth_drift=False, plot_sys_resp=False,
                       plot_track_results=False, plot_bright_results=False, rod_led_phase_correct=True,
                       filter_for_wierd_phases=True, exceptable_phase_insanity=0.1*np.pi, freq_err=0.01):

    """Returns a list of all acceptable system responses in selected direcotries"""

    measurements = select_filter_import_data(Iampl=Iampl, Ioffs=Ioffs, Ifreq=Ifreq, keyword_list=keyword_list)
    system_responses = []
    for measrmnt in measurements:
        sys_resp = freq_phase_ampl(measrmnt, freq_err=freq_err, plot_track_results=plot_track_results,
                                   plot_bright_results=plot_bright_results, bandfilter_smooth_drift=bandfilter_smooth_drift,
                                   rod_led_phase_correct=rod_led_phase_correct, exceptable_phase_insanity=exceptable_phase_insanity)

        if filter_for_wierd_phases:
            if not sys_resp.phase_sanity_check():
                system_responses.append(sys_resp)
            else:
                print("Measurement discarded because of insane phase!")
        else:
            system_responses.append(sys_resp)

    return system_responses
