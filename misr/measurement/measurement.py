import time
from os import path
from gvar import mean as gvalue

from ..utils.measurement_utils import cmd_set_led_current, cmd_set_current, cmd_set_frequency, cmd_start,\
    send_to_server, setup_serial, make_measurement_run_folder, check_current, check_frequency, get_hw_config
from ..analysis.import_trackdata import import_filepaths
from ..analysis.freq_and_phase_extract import freq_phase_ampl


def run_low_freq_ampl_cal(offset, ampl, freq=0.05, led_offset=0.09, led_ampl=0.03, wait_periods=4):
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


def run(offset, ampl, freqs, led_offset=0.09, led_ampl=0.03, pre_tracking_wait=5,
        max_measurement_time=420, num_periods=15, pixel_safety_margin=100, dynamic_amplitude=True):

    hw_conf = get_hw_config

    time_len = time_run(freqs, pre_tracking_wait, max_measurement_time, num_periods)
    print(f"The run will take no less than {round(time_len / 60)} minutes.")
    ans = str(input("Do you want to continue? [Y/n]"))
    if ans != "Y":
        exit()

    full_folder_path = make_measurement_run_folder()

    with setup_serial() as s:

        time.sleep(0.1)

        cmd_set_led_current(s, led_offset, led_ampl)
        cmd_set_current(s, offset, ampl)

        for freq in freqs:
            check_current(ampl, offset)
            check_frequency(freq, max_measurement_time)

            print("Current on!\nFrequency %.3f" % freq)
            cmd_set_frequency(s, freq)
            cmd_start(s)

            time.sleep(pre_tracking_wait)

            # After stationary conditions reached, the measurement can begin
            send_to_server("start_tracking")

            time.sleep(min(max_measurement_time, num_periods / freq))

            filename = f"offs{int(round(1000*offset))}_ampl{int(round(1000*ampl))}_freq{int(round(1000*freq))}.dat"
            full_filepath = path.join(full_folder_path, filename)
            send_to_server("stop_and_save_tracking " + full_filepath)

            time.sleep(2)

            if dynamic_amplitude:
                measurement = import_filepaths([full_filepath])[0]
                res = freq_phase_ampl(measurement)
                current_factor = (hw_conf["x_pixels"] - pixel_safety_margin - res.rod_mean) / gvalue(res.rod_ampl)
                ampl = min(hw_conf["max_current"] - offset, ampl*current_factor)

        print("Constant current!")
        cmd_set_current(s, offset, 0.00)
        cmd_set_frequency(s, 0.01)
        cmd_start(s)


def time_run(freqs, pre_tracking_wait, max_measurement_time, num_periods):
    timer = 0

    for freq in freqs:
        timer += pre_tracking_wait
        timer += min(max_measurement_time, num_periods / freq)
        timer += 2

    return timer

