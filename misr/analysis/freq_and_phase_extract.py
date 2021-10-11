import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
import math as m
from .import_trackdata import select_filter_import_data


def running_average(input_array, averaging_len):
    """
        This function calculates the running average, for arbitrary averageing_len, above the input_array.
        The i-th element in the output array is the average of elements with indexs: i, i+1, i+2, ..., i + averageing_len.
        The output array is of length = len(input_array) - averageing_len + 1, because no periodic boundary is imposed on the data!
    """
    return np.convolve(input_array, np.ones((averaging_len, ))/averaging_len, mode='valid')


def freq_phase_ampl(measrmnt, freq_err=0.1, plot_track_results=False, plot_bright_results=False, complex_drift=True):
    """ This function calculates the frequency of the vibration of the rod, the relative phase between
        the brightness modulation (current) and the response of the rod (position), and the amplitude of the rod response,
        from the results of rod tracking, and brightness logging.
        
        Inputs:
            measrmnt            --> The Measurement data object.
                measrmnt.trackData   ...    The data from rod tracking under the influence of changing magnetic fields.

                                            The trackData array is comprised of 4 columns:
                                            frameIdx, frameTime, rodEdgePos, brightnes

                                            Time is logged in seconds, and rodEdgePos in pixels. Brightness has arbitrary linear values.

            freq_err            --> The maximum relative discrepancy permitted between measured brihtness modulation and rod position modulation frequency.
                                    If the relative discrepancy is greater, the function returns a UserWarning

            complex_drift       --> A boolean value used to determine, weather to try to consider more complex then just linear drift in the rod position fit.

            plot_track_results  --> A boolean value used to determine, weather to plot the results from tracking rod's position
            
            plot_bright_results --> A boolean value used to determine, weather to plot the results from brightnes analisys


        Returns:
            The function doesnt return anything. It changes the measurement object, and writes the following fields:

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

    # This is just an educated guess, average of 100 pixel values gives an error of a pixel_error/sqrt(100)
    sigmas_bright = np.ones((measrmnt.numFrames, )) * 0.1

    try:
        bright_coefs, bright_cov = curve_fit(sinusoid_w_drift, measrmnt.times, measrmnt.brights, sigma=sigmas_bright, absolute_sigma=True,
            bounds=([0.5*(max_bright - min_bright)/2, measrmnt.Ifreq*0.8, -np.pi, min_bright, -(max_bright - min_bright)/measrmnt.timeLength],
                    [1.2*(max_bright - min_bright)/2., measrmnt.Ifreq*1.2, np.pi, max_bright, (max_bright - min_bright)/measrmnt.timeLength]))
    except RuntimeError:
        print("The least squares minimization for brightness fitting failed!")

    if plot_bright_results is True:
        # Display brightness result and fit
        plt.plot(measrmnt.times, sinusoid_w_drift(measrmnt.times, *bright_coefs), label=r'Fit')
        plt.plot(measrmnt.times, measrmnt.brights, label=r'Track result')
        plt.title("BRIGHT - Freq: {0}".format(measrmnt.Ifreq))
        plt.legend()
        plt.show()
    
    # -----------------------------------------------------------------------------------------

    # Fitting rod position data ---------------------------------------------------------------
    min_pos = np.min(measrmnt.positions)
    max_pos = np.max(measrmnt.positions)
    
    sigmas_pos = np.ones((measrmnt.numFrames, )) * 0.5      # This is just an educated guess, one pixel precision is assumed

    try:
        # First: fit just for the exact frequency
        pos_coefs, pos_cov = curve_fit(sinusoid_w_drift, measrmnt.times, measrmnt.positions, sigma=sigmas_pos, absolute_sigma=True,
            bounds=([0.5*(max_pos - min_pos)/2, measrmnt.Ifreq*0.8, bright_coefs[2] - np.pi, min_pos, -(max_pos - min_pos)/measrmnt.timeLength],
                    [1.2*(max_pos - min_pos)/2., measrmnt.Ifreq*1.2, bright_coefs[2], max_pos, (max_pos - min_pos)/measrmnt.timeLength]))

        # Second: correct for the undesired, complex drift
        averaging_len = int(measrmnt.numFrames / (pos_coefs[1] * measrmnt.timeLength))
        averaging_len_good = False
        if 2 < averaging_len < 0.5*measrmnt.numFrames:      # Must be > than 2 bc otherwise the measrmnt.trackData with the offsets would be 0 long!
            averaging_len_good = True
            drift = running_average(measrmnt.positions, averaging_len)
            # print("Drift:", np.std(drift))

        # Only permitted to continue if averaging_len is within reason, and the complex drift is sought to be considered
        if averaging_len_good and complex_drift is True:
            # Third: Remove drift (from center of data)
            drift_start_offset = averaging_len//2
            drift_stop_offset = -(averaging_len - averaging_len//2) + 1
            # print("DEBUG:", measrmnt.trackData, drift)
            # print("drift_start_offset= {}, drift_stop_offset={}".format(drift_start_offset, drift_stop_offset))

            driftless_pos_data = measrmnt.positions[drift_start_offset:drift_stop_offset] - drift
            driftless_time = measrmnt.times[drift_start_offset:drift_stop_offset]

            # Fourth: Calculate new fitting boundaries and sigmas
            min_pos = np.min(driftless_pos_data)
            max_pos = np.max(driftless_pos_data)

            # This is just an educated guess, one pixel precision is assumed (drift worsens the error)
            sigmas_pos = np.ones((len(driftless_pos_data), )) * 1.

            # Finally: Refit the sinusoid to the driftless_data
            pos_coefs, pos_cov = curve_fit(sinusoid_w_drift, driftless_time, driftless_pos_data, sigma=sigmas_pos, absolute_sigma=True,
                bounds=([0.5*(max_pos - min_pos)/2, measrmnt.Ifreq*0.8, bright_coefs[2] - np.pi, min_pos, -(max_pos - min_pos)/measrmnt.timeLength],
                        [1.2*(max_pos - min_pos)/2., measrmnt.Ifreq*1.2, bright_coefs[2], max_pos, (max_pos - min_pos)/measrmnt.timeLength]))

            if plot_track_results == True:
                # Display rod position track result and fit
                plt.plot(driftless_time, sinusoid_w_drift(driftless_time, *pos_coefs) + drift, label=r'Fit')
                plt.plot(measrmnt.times, measrmnt.positions, label=r'Track result')
                plt.title("Freq: {0}".format(measrmnt.Ifreq))
                plt.legend()
                plt.show()
        
        else:
            if plot_track_results == True:
                # Display rod position track result and fit
                plt.plot(measrmnt.times, sinusoid_w_drift(measrmnt.times, *pos_coefs), label=r'Fit')
                plt.plot(measrmnt.times, measrmnt.positions, label=r'Track result')
                plt.title("Freq: {0}".format(measrmnt.Ifreq))
                plt.legend()
                plt.show()

    except RuntimeError:
        print("The least squares minimization for position fitting failed!")
    # -----------------------------------------------------------------------------------------
    
    # Check for frequency discrepancy
    if m.fabs((bright_coefs[1] - pos_coefs[1])/bright_coefs[1]) >= freq_err :
        warnings.warn("Relativna razlika v frekvencah dobljenih iz brightness data in position data je vecja od {0}".format(freq_err) +
        "\nBrightness freq: {0}\nPosition freq: {1}".format(bright_coefs[1], pos_coefs[1]))

    # Write the results to the measurement object
    measrmnt.rod_freq = pos_coefs[1]
    measrmnt.rod_freq_err = m.sqrt(pos_cov[1, 1])
    measrmnt.rod_ampl = pos_coefs[0]
    measrmnt.rod_ampl_err = m.sqrt(pos_cov[0, 0])
    measrmnt.rod_phase = pos_coefs[2]
    measrmnt.rod_phase_err = m.sqrt(bright_cov[2, 2]) + m.sqrt(pos_cov[2, 2])     # PREDPOSTAVKA DA KR SEŠTEJEŠ SIGME
