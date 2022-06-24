import time
from os import path

from ..utils.measurement_utils import cmd_set_led_current, cmd_set_current, cmd_set_frequency,\
    cmd_start, send_to_server, setup_serial, make_measurement_run_folder,\
    check_freqs, get_hw_config, set_framerate, ask_do_you_want_to_continue,\
    calc_new_dynamic_ampl, print_prerun_checklist, sleep_and_record_comments


def run_low_freq_ampl_cal(offset, ampl, freq=0.05, led_offset=0.09, led_ampl=0.03, wait_periods=4):
    """ Runs low frequency oscillations, for setting up the field of view, camera settings and amplitude setting.
        Doesn't record or track the rod, just runs the coils. Doesn't return anything.

        Attributes:
            - offset            Current offset on coils, in amperes.
            - ampl              Current amplitude on coils, in amperes.
            - freq              The frequency of slow oscillations, in Hz.
            - led_offset        Current offset on led, in amperes.
            - led_ampl          Current ampl on led, in amperes. Used for
                                relative phase measurement between coil and rod response.
            - wait_periods      Number of periods the run is active
    """
    with setup_serial() as s:
        time.sleep(0.1)

        print("Current on!")
        cmd_set_led_current(s, led_offset, led_ampl)
        cmd_set_current(s, offset, ampl)
        cmd_set_frequency(s, freq)

        cmd_start(s)

        time.sleep(wait_periods / freq)

        cmd_set_current(s, offset, 0.00)
        cmd_start(s)
        print("Constant current!")


def run(offset, ampl, freqs, led_offset=0.09, led_ampl=0.03, pre_tracking_wait=5, post_tracking_wait=2,
        max_measurement_time=420, num_periods=15, min_num_periods=4, pixel_safety_margin=100, dynamic_amplitude=True,
        datapoints_per_period=20, dynamic_framerate=True, min_framerate=None):

    """ Runs the measurement run for all requested frequencies, at starting amplitude and offset settings.
        The measurement run consists of:
            - Checking the input frequencies
            - Predicting the time of completion
            - Making a new measurement folder
            - Running the coils for the experiment
            - Recording and tracking the rod
            - Possibly modifying the frequencies and amplitudes mid run

        Doesn't return anything, but writes the tracking results to output files in a new subdirectory of the measurement folder.
        The name of the output subdirectory is chosen in this function, the name of the files is subject to naming convention:
            f"offs{round(1000 * offset)}_ampl{round(1000 * ampl)}_freq{round(1000 * freq)}.dat"

        Attributes:
            - offset        Starting current offset on coils, in amperes.
                            May change later if dynamic_amplitude is True.
            - ampl          Starting current amplitude on coils, in amperes.
                            May change later if dynamic_amplitude is True.
            - freq              The array of frequencies, at which the response is to be probed. In units of Hz.
            - led_offset        Current offset on led, in amperes.
            - led_ampl          Current ampl on led, in amperes. Used for
                                relative phase measurement between coil and rod response.
            - pre_tracking_wait     Wait time in seconds between change of coil frequency or amplitude and start of tracking.
            - post_tracking_wait    Wait time in seconds after stop of tracking and new change of coil frequency or amplitude.
                                    Allows for mid run amplitude tuning.
            - max_measurement_time      Maximum time allowed for tracking the rod at one frequency. In units of seconds.
            - num_periods               The number of oscillations to be tracked and recorded for each frequency, IF
                                        the max_measurement_time permits it for that frequency.
            - min_num_periods           The minimum number of periods to be tracked for each frequency. If the maximum
                                        allowed time is to short even for this condition to be satisfied, the program
                                        warns the user, and asks if the user wants to continue with the run.
            - pixel_safety_margin       If dynamic_amplitude is True, this value is used to calculate
                                        the new current amplitude, so as to oscillate the rod between
                                        the pixel ranges of [pixel_safety_margin, end - pixel_safety_margin]
            - dynamic_amplitude         This option allows for approximation of oscillation amplitude mid run, and
                                        dynamic change of the current amplitude for maximum SNR in the rod oscillation.
            - datapoints_per_period     If dynamic_framerate is True, this number indicates the desired number of
                                        recorded and tracked frames per one period of a specific frequency. Because
                                        this is a function of frequency, the program needs to be able to change the
                                        sampling framerate.
            - dynamic_framerate         This option allows for change of framerate for each frequency, so as to satisfy
                                        the request for a specific number of data points per period of oscillation.
                                        FOR THIS TO WORK, the framerate must be set to "On" in the camera settings.
            - min_framerate             Minimum framerate of the recording. If the framerate is to low the shutter speed
                                        rises and as a result the video becomes washed out.
    """

    print_prerun_checklist()

    hw_conf = get_hw_config()
    max_framerate = float(input("Please input the maximum framerate: "))
    if min_framerate is None:
        min_framerate = float(input("Please input the minimum framerate: "))

    check_freqs(freqs, max_measurement_time, min_num_periods, datapoints_per_period, max_framerate)
    time_run(freqs, pre_tracking_wait, post_tracking_wait, max_measurement_time, num_periods)

    full_folder_path = make_measurement_run_folder()

    with setup_serial() as s:

        time.sleep(0.1)
        cmd_set_led_current(s, led_offset, led_ampl)
        cmd_set_current(s, offset, ampl)

        with open(path.join(full_folder_path, "comments.txt"), "w") as comments_file:
            for freq in freqs:
                filename = f"offs{int(round(1000*offset))}_ampl{int(round(1000*ampl))}_freq{int(round(1000*freq))}.dat"
                full_filepath = path.join(full_folder_path, filename)

                print("Current on!\nFrequency %.3f" % freq)
                cmd_set_frequency(s, freq)
                if dynamic_framerate:
                    set_framerate(max(min_framerate, min(datapoints_per_period * freq, max_framerate)))
                cmd_start(s)

                time.sleep(pre_tracking_wait)
                send_to_server("start_tracking")

                sleep_time = min(max_measurement_time, num_periods / freq)
                # user_exit = sleep_and_record_comments(sleep_time, comments_file, filename, cuttoff_time=0.01)
                time.sleep(sleep_time)

                send_to_server(f"stop_and_save_tracking {full_filepath}")

                if user_exit:
                    time.sleep(1.)
                    ask_do_you_want_to_continue("User typed exit() or quit() mid-run!", s=s)

                if not dynamic_amplitude:
                    time.sleep(post_tracking_wait)
                else:
                    start_time = time.perf_counter()
                    time.sleep(post_tracking_wait / 2.0)
                    ampl = calc_new_dynamic_ampl(offset, ampl, full_filepath, pixel_safety_margin, hw_conf)
                    cmd_set_current(s, offset, ampl)
                    stop_time = time.perf_counter()
                    time.sleep(max(0.01, post_tracking_wait - (stop_time - start_time)))

        print("Constant current!")
        cmd_set_current(s, offset, 0.00)
        cmd_set_frequency(s, 0.01)
        if dynamic_framerate:
            set_framerate(max_framerate)
        cmd_start(s)


def time_run(freqs, pre_tracking_wait, post_tracking_wait, max_measurement_time, num_periods):
    """ This function roughly approximates the time (in seconds), the measurement run will take.
        Attributes:
            - pre_tracking_wait     Wait time in seconds between change of coil frequency or amplitude and start of tracking.
            - post_tracking_wait    Wait time in seconds after stop of tracking and new change of coil frequency or amplitude.
                                    Allows for mid run amplitude tuning.
            - max_measurement_time      Maximum time allowed for tracking the rod at one frequency. In units of seconds.
            - num_periods               The number of oscillations to be tracked and recorded for each frequency, IF
                                        the max_measurement_time permits it for that frequency.

        The function returns the total approximated time of the measurement run in seconds.
    """

    timer = 0

    for freq in freqs:
        timer += pre_tracking_wait
        timer += min(max_measurement_time, num_periods / freq)
        timer += post_tracking_wait

    print(f"The run will take no less than {round(timer / 60)} minutes.")
    ask_do_you_want_to_continue()
