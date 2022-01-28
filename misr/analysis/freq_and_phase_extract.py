import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
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


def freq_phase_ampl(measrmnt, freq_err=0.1, plot_track_results=False,
    plot_bright_results=False, complex_drift=True, rod_led_phase_correct=True, exceptable_phase_insanity=0.1*np.pi,
    ignore_extreme_rod_positions=False):
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

            complex_drift       --> A boolean value used to determine, weather to try to consider more complex then just linear drift in the rod position fit.

            plot_track_results  --> A boolean value used to determine, weather to plot the results from tracking rod's position
            
            plot_bright_results --> A boolean value used to determine, weather to plot the results from brightnes analisys

            ignore_extreme_rod_positions  --> If set to True, the fitting of position will ignore all 0 rod position values


        Returns:
            The function returns a SingleResult class object with the following fields:

            rod_freq        -->     The frequencie of the vibration
            rod_freq_err    -->     The absolute error in frequencies
            rod_phase       -->     The relative phase (in radians) between the position and curent modulation (brightness).
            rod_phase_err   -->     The error of relative phase
            rod_ampl        -->     The amplitude of the rod position modulation (in pixels)
            rod_ampl_err    -->     The absolute error in the amplitude of the rod position modulation

        """

    # Fitting function (allowing for linear drift)
    def sinusoid_w_drift(t, A, freq, phase, const, speed):
        """
            The function used for fitting measured rod position and brightness data!
        """
        return A * np.sin(2*np.pi*freq * t + phase) + const + speed * t

    # Fitting brightness data ------------------------------------------------------------------
    min_bright = np.min(measrmnt.brights)
    max_bright = np.max(measrmnt.brights)

    low_bound_bright = gvalue([0.5 * (max_bright - min_bright) / 2., measrmnt.Ifreq * 0.8, 0.,
                               min_bright, -(max_bright - min_bright) / measrmnt.timeLength])
    high_bound_bright = gvalue([1.2 * (max_bright - min_bright) / 2., measrmnt.Ifreq * 1.2, 2 * np.pi,
                                max_bright, (max_bright - min_bright) / measrmnt.timeLength])

    try:
        bright_coefs, bright_cov = curve_fit(sinusoid_w_drift, gvalue(measrmnt.times), gvalue(measrmnt.brights),
                                             sigma=sdev(measrmnt.brights), absolute_sigma=True,
                                             bounds=(low_bound_bright, high_bound_bright))
    except RuntimeError:
        print("The least squares minimization for brightness fitting failed!")

    bright_coefs = gvar(bright_coefs, bright_cov)

    if plot_bright_results is True:
        # Display brightness result and fit
        best_fit = sinusoid_w_drift(gvalue(measrmnt.times), *bright_coefs)
        plt.plot(gvalue(measrmnt.times), gvalue(best_fit), label=r'Fit')
        plt.fill_between(gvalue(measrmnt.times), gvalue(best_fit) - sdev(best_fit), gvalue(best_fit) + sdev(best_fit), alpha = 0.3)

        plt.errorbar(gvalue(measrmnt.times), gvalue(measrmnt.brights),
                     yerr=sdev(measrmnt.brights), label=r'Track result')
        plt.title("BRIGHT - Freq: {0}".format(measrmnt.Ifreq))
        plt.legend()
        plt.show()
    # -----------------------------------------------------------------------------------------

    # Fitting rod position data ---------------------------------------------------------------

    # GRDI GRDI GRDI GRDI TRY
    # TODO: --> TOLE MORAŠ LEPŠE ZAPISAT!!!!! -----------------------------------------------------------------
    try:
        # TODO: BETTER FITTING STRATEGY!!!!!
        # TRY FILTERING WITH FFT
        amplitudes = np.fft.rfft(gvalue(measrmnt.positions))
        freqs = np.fft.rfftfreq(len(measrmnt.positions), d=gvalue(measrmnt.times[1] - measrmnt.times[0]))
        print(measrmnt.Ifreq)
        print(freqs)
        plt.plot(freqs, np.square(np.abs(amplitudes)))
        plt.show()
        # JUST A TRY


        if ignore_extreme_rod_positions:
            nonzero_elems = np.flatnonzero(measrmnt.positions)
            fit_times = gvalue(measrmnt.times)[nonzero_elems]
            fit_positions = gvalue(measrmnt.positions)[nonzero_elems]
            fit_sigmas = sdev(measrmnt.positions)[nonzero_elems]
        else:
            fit_times = gvalue(measrmnt.times)
            fit_positions = gvalue(measrmnt.positions)
            fit_sigmas = sdev(measrmnt.positions)

        min_pos = np.min(fit_positions)
        max_pos = np.max(fit_positions)

        # TODO: FIX THIS HORRIBLE CODE BLOCK!!!!

        low_bound_pos = gvalue([0.5 * (max_pos - min_pos) / 2., measrmnt.Ifreq * 0.8, 0.,
                                min_pos, -(max_pos - min_pos) / measrmnt.timeLength])
        high_bound_pos = gvalue([1.2*(max_pos - min_pos)/2., measrmnt.Ifreq*1.2, 2*np.pi,
                             max_pos, (max_pos - min_pos)/measrmnt.timeLength])

        # First: fit just for the exact frequency, disregarding phase information
        pos_coefs, pos_cov = curve_fit(sinusoid_w_drift, fit_times, fit_positions,
                                       sigma=fit_sigmas, absolute_sigma=True,
                                       bounds=(low_bound_pos, high_bound_pos))

        pos_coefs = gvar(pos_coefs, pos_cov)

        # Second: correct for the undesired, complex drift
        averaging_len = int(measrmnt.numFrames / gvalue(pos_coefs[1] * measrmnt.timeLength))
        averaging_len_good = False
        if 2 < averaging_len < 0.5*measrmnt.numFrames:      # Must be > than 2 bc otherwise the measrmnt.trackData with the offsets would be 0 long!
            averaging_len_good = True
            drift = running_average(measrmnt.positions, averaging_len)
            # print("Drift:", np.std(drift))

        # Only permitted to continue if averaging_len is within reason, and the complex drift is sought to be considered
        # TODO: WHEN IGNORE EXTREME POSITIONS is fixed, remove the added pogoj!
        if averaging_len_good and complex_drift is True and not ignore_extreme_rod_positions:
            # Calculate drif
            drift_start_offset = averaging_len//2
            drift_stop_offset = -(averaging_len - averaging_len//2) + 1

            # Third: Remove drift (from center of data)
            driftless_pos_data = measrmnt.positions[drift_start_offset:drift_stop_offset] - drift
            driftless_time = measrmnt.times[drift_start_offset:drift_stop_offset]

            # Fourth: Calculate new fitting boundaries and sigmas
            min_pos = np.min(driftless_pos_data)
            max_pos = np.max(driftless_pos_data)

            low_bound_pos = gvalue([0.5*(max_pos - min_pos)/2., measrmnt.Ifreq*0.8,
                                    bright_coefs[2] - 2*np.pi - exceptable_phase_insanity,
                                    min_pos, -(max_pos - min_pos)/measrmnt.timeLength])
            high_bound_pos = gvalue([1.2*(max_pos - min_pos)/2., measrmnt.Ifreq*1.2,
                                     bright_coefs[2] + exceptable_phase_insanity,
                                     max_pos, (max_pos - min_pos)/measrmnt.timeLength])

            # TODO: INCORPORATE CONVOLUTING over FITTING of "intact" parts of trajectories
            # # Possibly ignore extreme rod positions   
            # if ignore_extreme_rod_positions:
            #     nonzero_elems = np.flatnonzero(measrmnt.positions[drift_start_offset:drift_stop_offset]),
            #     fit_times = gvalue(measrmnt.times)[nonzero_elems]
            #     fit_positions = gvalue(measrmnt.positions)[nonzero_elems]
            #     fit_sigmas = sdev(measrmnt.positions)[nonzero_elems]
            # else:
            #     fit_times = gvalue(measrmnt.times)
            #     fit_positions = gvalue(measrmnt.positions)
            #     fit_sigmas = sdev(measrmnt.positions)
            # THE INCORPORATE THIS INTO THE FITTING PROCEDURE!!!
            # pos_coefs, pos_cov = curve_fit(sinusoid_w_drift, fit_times, fit_positions,
            #                             sigma=fit_sigmas, absolute_sigma=True,
            #                             bounds=(low_bound_pos, high_bound_pos))

            # Finally: Refit the sinusoid to the driftless_data
            pos_coefs, pos_cov = curve_fit(sinusoid_w_drift, gvalue(driftless_time), gvalue(driftless_pos_data),
                                        sigma=sdev(driftless_pos_data), absolute_sigma=True,
                                        bounds=(low_bound_pos, high_bound_pos))

            pos_coefs = gvar(pos_coefs, pos_cov)

            if plot_track_results == True:
                # Display rod position track result and fit
                best_fit = sinusoid_w_drift(gvalue(driftless_time), *pos_coefs) + drift
                plt.plot(gvalue(driftless_time), gvalue(best_fit), label=r'Fit')
                plt.fill_between(gvalue(driftless_time), gvalue(best_fit) - sdev(best_fit),
                                 gvalue(best_fit) + sdev(best_fit), alpha=0.3)

                plt.errorbar(gvalue(measrmnt.times), gvalue(measrmnt.positions),
                             yerr=sdev(measrmnt.positions), label=r'Track result')
                plt.title("Freq: {0}".format(measrmnt.Ifreq))
                plt.legend()
                plt.show()
        
        else:
            if plot_track_results == True:
                # Display rod position track result and fit
                best_fit = sinusoid_w_drift(gvalue(measrmnt.times), *pos_coefs)
                plt.plot(gvalue(measrmnt.times), gvalue(best_fit), label=r'Fit')
                plt.fill_between(gvalue(measrmnt.times), gvalue(best_fit) - sdev(best_fit),
                                 gvalue(best_fit) + sdev(best_fit), alpha=0.3)

                plt.errorbar(gvalue(measrmnt.times), gvalue(measrmnt.positions),
                             yerr=sdev(measrmnt.positions), label=r'Track result')
                plt.title("Freq: {0}".format(measrmnt.Ifreq))
                plt.legend()
                plt.show()

    except RuntimeError:
        print("The least squares minimization for position fitting failed!")
    # -----------------------------------------------------------------------------------------
    
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


def select_and_analyse(Iampl="*", Ioffs="*", Ifreq="*", keyword_list=[], complex_drift=True, plot_sys_resp=False,
                       plot_track_results=False, plot_bright_results=False, rod_led_phase_correct=True,
                       filter_for_wierd_phases=True, exceptable_phase_insanity=0.1*np.pi, freq_err=0.01):

    """Returns a list of all acceptable system responses in selected direcotries"""

    measurements = select_filter_import_data(Iampl=Iampl, Ioffs=Ioffs, Ifreq=Ifreq, keyword_list=keyword_list)
    system_responses = []
    for measrmnt in measurements:
        sys_resp = freq_phase_ampl(measrmnt, freq_err=freq_err, plot_track_results=plot_track_results,
                                   plot_bright_results=plot_bright_results, complex_drift=complex_drift,
                                   rod_led_phase_correct=rod_led_phase_correct, exceptable_phase_insanity=exceptable_phase_insanity)

        if filter_for_wierd_phases:
            if not sys_resp.phase_sanity_check():
                system_responses.append(sys_resp)
            else:
                print("Measurement discarded because of insane phase!")
        else:
            system_responses.append(sys_resp)

    return system_responses
