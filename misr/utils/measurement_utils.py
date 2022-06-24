import time, socket, serial, re, sys
from select import select
from warnings import warn
from datetime import datetime
from os import path, mkdir
from .config import get_config
from functools import cache
from ..analysis.freq_and_phase_extract import freq_phase_ampl
from ..analysis.import_trackdata import import_filepaths
from gvar import mean as gvalue
from ..utils.save_and_get_results import save_response_to_txt


@cache
def get_hw_config():
    """
    Fetch the hardware configuration of the measurement setup like: internet serial port name, logfile name,
    serial baudrate, server port, server address, max current for continuouse operation and image width.

    Results are returned in a dictionary with keys:
        - serial_port       Designates the serial port for communication with the coil current control
        - logfilename       Full name of the log file.
        - bd                Serial baudrate
        - x_pixels          Full image width (maximum allowed rod oscillation amplitude
        - server_port       The server port for communication with camera software
        - server_address    The server address for camera communications
        - max_current       Maximum allowed constant current trough coils, in amperes.
        - pixel_size        Pixel size expressed in m/pix. Is dependant on resolution, and microscope.
    """
    hw_config_filepath = get_config()["hw"]

    with open(hw_config_filepath, "r") as hw_config_file:
        file_contents = "\n".join(hw_config_file.readlines())

        serial_port = re.search("PORT='.*'", file_contents)[0][6:-1]
        logfilename = re.search("LOG='.*'", file_contents)[0][5:-1]
        baudrate = int(re.search("BAUDRATE='.*'", file_contents)[0][10:-1])
        server_port = int(re.search("SERVER_PORT='.*'", file_contents)[0][13:-1])
        server_address = re.search("SERVER_ADDRESS='.*'", file_contents)[0][16:-1]
        max_current = float(re.search("MAX_CURRENT='.*A'", file_contents)[0][13:-2])
        x_pixels = int(re.search("X_PIXEL_COUNT='.*'", file_contents)[0][15:-1])
        pixel_size = float(re.search("PIXEL_SIZE='.*'", file_contents)[0][12:-1])

    return {"serial_port": serial_port, "logfilename": logfilename, "bd": baudrate, "x_pixels": x_pixels,
            "server_port": server_port, "server_address": server_address, "max_current": max_current, "pixel_size": pixel_size}


def setup_serial():
    """ Open serial port, set up communication, and return the serial.Serial object. Can work as an context manager! """
    hw_config = get_hw_config()
    return serial.Serial(port=hw_config["serial_port"], timeout=1, baudrate=hw_config["bd"])


def send_serial_command(s, cmd):
    """ Sends serial commands to the coil current controller and recieves the response. Logs both.
    Gets a serial instance s and a command cmd."""
    logfilename = get_hw_config()["logfilename"]
    with open(logfilename, "a") as logfile:
        print(f"{time.strftime('%Y-%m-%d %H-%M-%S')} SENDING TO SERIAL: {cmd}", file=logfile)
        s.write(cmd)
        time.sleep(0.1)
        line = s.readline()
        print(f"{time.strftime('%Y-%m-%d %H-%M-%S')} READING FROM SERIAL: {line}", file=logfile)


def get_calibrated_current_value(channel, currentA):
    """ Calculates the current (amperes) to setting (arb. num.) conversion for each coil/channel."""
    if channel == 1:
        return 0.5224 * currentA + 0.001412
    elif channel == 2:
        return 0.4427 * currentA + 0.001495
    else:
        return 0.0


def cmd_stop(s):
    """ Send stop command. Gets a serial instance s."""
    send_serial_command(s, b'STOP\r\n')


def cmd_start(s):
    """ Send start command. Gets a serial instance s"""
    send_serial_command(s, b'START\r\n')


def cmd_set_frequency(s, frequency):
    """ Send frequency change command. Gets a serial instance s and the frequency."""
    send_serial_command(s, b'FREQ=%.3f\r\n' % frequency)


def cmd_set_current(s, offsetA, amplitudeA):
    """ Set current on both coils. Gets a serial instance s and offset and amplitude currents.
    All current values are passed in amperes."""
    check_current(amplitudeA, offsetA, s=s)

    maxCurrent = offsetA + amplitudeA
    minCurrent = max(offsetA - amplitudeA, 0)

    ch1max = get_calibrated_current_value(1, maxCurrent)
    ch1min = get_calibrated_current_value(1, minCurrent)
    ch2max = get_calibrated_current_value(2, maxCurrent)
    ch2min = get_calibrated_current_value(2, minCurrent)

    send_serial_command(s, b'OFFCH1=%.3f\r\n' % ((ch1max + ch1min) / 2.0))
    send_serial_command(s, b'OFFCH2=%.3f\r\n' % ((ch2max + ch2min) / 2.0))
    send_serial_command(s, b'AMPCH1=%.3f\r\n' % ((ch1max - ch1min) / 2.0))
    send_serial_command(s, b'AMPCH2=%.3f\r\n' % ((ch2max - ch2min) / 2.0))


def cmd_set_current_on_coil(s, channel, offsetA, amplitudeA):
    """ Set current on each coil separately. Gets a serial instance s, offset and amplitude currents and the channel id.
    All current values are passed in amperes. Channel id is either 1 or 2."""
    check_current(amplitudeA, offsetA, s=s)

    maxCurrent = offsetA + amplitudeA
    minCurrent = max(offsetA - amplitudeA, 0)

    chmax = get_calibrated_current_value(channel, maxCurrent)
    chmin = get_calibrated_current_value(channel, minCurrent)

    if channel == 1:
        send_serial_command(s, b'OFFCH1=%.3f\r\n' % ((chmax + chmin) / 2.0))
        send_serial_command(s, b'AMPCH1=%.3f\r\n' % ((chmax - chmin) / 2.0))
    elif channel == 2:
        send_serial_command(s, b'OFFCH2=%.3f\r\n' % ((chmax + chmin) / 2.0))
        send_serial_command(s, b'AMPCH2=%.3f\r\n' % ((chmax - chmin) / 2.0))
    else:
        return None


def cmd_set_led_current(s, offsetA, amplitudeA):
    """ Send led offset and amplitude settings. Gets serial instantce s and offset and amplitude currents.
    All currents passes in amperes."""
    send_serial_command(s, b'OFFLED=%.3f\r\n' % offsetA)
    send_serial_command(s, b'AMPLED=%.3f\r\n' % amplitudeA)


def send_to_server(message):
    """ Sends a message trough TCP/IP socket to the software controlling the camera. Logs the message sent and
    any response recieved back. Server address, port and log filename are gotten from the hardware config file.
    """
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Get the address and port where the server is listening
    hw_conf = get_hw_config()
    server_address = (hw_conf["server_address"], hw_conf["server_port"])

    with open(hw_conf["logfilename"], "a") as logfile:
        try:
            # Connect to server
            sock.connect(server_address)

            # Send data
            print(f"{time.strftime('%Y-%m-%d %H-%M-%S')} SENDING TO SERVER: {message}", file=logfile)
            sock.send(message.encode())

            # Look for the response
            data = sock.recv(1024)
            print(f"{time.strftime('%Y-%m-%d %H-%M-%S')} RECEIVED FROM SERVER {data}", file=logfile)

        except Exception as e:
            print(f"Exception {e.__class__} OCCURED!", file=logfile)
            warn(f"Exception {e.__class__} OCCURED!")

    # Close the socket - WHY WAS IT COMMENTED OUT? MAYBE THIS CAN CAUSE PROBLEMS - I WILL TRY WITH IT ENABLED AND SEE
    # PAZI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sock.close()


def set_framerate(framerate):
    """ Set framerate of the camera. """
    send_to_server('SET_FRAMERATE %.3f' % framerate)


def enable_led(led_offset=0.09, led_ampl=0.0):
    """ Enables the LED with specified offsets and amplitudes."""
    with setup_serial() as s:
        time.sleep(0.1)
        cmd_set_led_current(s, led_offset, led_ampl)
        cmd_set_frequency(s, 0.1)
        time.sleep(0.1)
        cmd_start(s)


def disable_all(s=None):
    """ Disable all outputs!
    If serial.Serial object is present in s, then the method will use the allready set up instance provided! """
    if isinstance(s, serial.Serial):
        time.sleep(0.1)
        cmd_set_led_current(s, 0., 0.)
        cmd_stop(s)
    else:
        with setup_serial() as s:
            time.sleep(0.1)
            cmd_set_led_current(s, 0., 0.)
            cmd_stop(s)


def disable_all_coils(s=None):
    """ Disable all coils!
    If serial.Serial object is present in s, then the method will use the allready set up instance provided! """
    if isinstance(s, serial.Serial):
        time.sleep(0.1)
        cmd_stop(s)
    else:
        with setup_serial() as s:
            time.sleep(0.1)
            cmd_stop(s)


def make_measurement_run_folder():
    """ Creates a measurement run folder, based on current date, and a subfolder based on the specified name.
    Also adds the measurement configuration file to the folder, so the measurement class knows to find it. """
    dt = datetime.now()
    measurement_folder = get_config()["meas"]
    date_folder_path = path.join(measurement_folder, dt.strftime('%Y-%m-%d'))

    if not path.isdir(date_folder_path):
        mkdir(date_folder_path)

    print("You are starting a new run!")
    rod_id = int(input("Please input the rod id: "))
    tub_id = int(input("Please input the tub id: "))

    print("How would you like to name the folder?")
    while True:
        folder_name = str(input())

        if not path.isdir(path.join(date_folder_path, folder_name)):
            break
        print("Folder name already exists! Choose a new one!")

    full_folder_path = path.join(date_folder_path, folder_name)
    mkdir(full_folder_path)

    # Add measurement config file
    with open(path.join(full_folder_path, ".meas_config"), "w") as meas_conf_file:
        print(f"X_PIXEL_COUNT='{get_hw_config()['x_pixels']}'", file=meas_conf_file)
        print(f"PIXEL_SIZE='{get_hw_config()['pixel_size']:.10f}'", file=meas_conf_file)
        print(f"ROD_ID='{rod_id}'", file=meas_conf_file)
        print(f"TUB_ID='{tub_id}'", file=meas_conf_file)

    return full_folder_path


def check_current(ampl, offs, s=None):
    """ Checks the validity of proposed current updates. If a discrepancy is discovered,
    a user intervention in form of a warning and a question to continue is raised.
    If the funciton is called during a run sestion, one MUST provide serial.Serial instance in use,
    as the attribute s. """
    max_curr = get_hw_config()["max_current"]
    if (ampl + offs > max_curr) or (offs - ampl < 0.0):
        ask_do_you_want_to_continue(f"Some method wants to set the max current to {ampl + offs} but maximum allowed is {max_curr}!", s=s)


def check_freqs(freqs, max_time, min_num_periods, min_datapoints_per_period, max_framerate):
    """ Checks for 2 things:
            - Maximum measurement time allows for measurement of at least a minimum number of periods for each frequency.
            - Maximum framerate allows for measurement of at least a minimum number of data points per period.
        If discrepancy is discovered, a user intervention in form of a warning and a question to continue is raised.

        Attributes:
            - freqs                         Measurement frequencies
            - max_time                      Maximum measurement time for each frequency
            - min_num_periods               The minimum number of
            - min_datapoints_per_period     The minimum number of data points to be gathered per period for each freq.
            - max_framerate                 The maximum framerate.

    """
    any_warnings = False
    for freq in freqs:
        if max_time * freq < min_num_periods:
            warn(f"For frequency {freq} there will be only {max_time * freq} periods instead of {min_num_periods}!")
            any_warnings = True
        if min_datapoints_per_period > max_framerate / freq:
            warn(f"For frequency {freq} there will be only {max_framerate / freq} of datapoints per period instead of {min_datapoints_per_period}!")
            any_warnings = True

    if any_warnings:
        ask_do_you_want_to_continue()


def ask_do_you_want_to_continue(warning=None, s=None):
    """ Raises a warning with a user defines message, and asks for user intervention to continue.
    If the funciton is called during a run sestion, one MUST provide serial.Serial instance in use,
    as the attribute s. """
    if warning is not None:
        warn(warning)
    ans = str(input("Do you want to continue? [y/N]"))
    if ans.upper() == "N":
        disable_all_coils(s)
        exit()


def calc_new_dynamic_ampl(offset, ampl, full_filepath, pixel_safety_margin, hw_conf):
    measurement = import_filepaths([full_filepath])[0]
    res = freq_phase_ampl(measurement)
    # Ime za shranit file!
    folder_name = path.split(full_filepath)[0]
    experiment_foldername = path.split(folder_name)[1]
    date_foldername = path.split(path.split(folder_name)[0])[1]
    full_export_filename = path.join(folder_name, f"{date_foldername}_{experiment_foldername}.txt")
    save_response_to_txt(res, full_export_filename)    # Zraven še shranimo!
    # Izračuna faktorje in vse te stvari
    current_factor = (hw_conf["x_pixels"] - pixel_safety_margin - res.rod_mean) / gvalue(res.rod_ampl)
    ampl = max(0.001, min(hw_conf["max_current"] - offset, ampl * current_factor))
    print(f"New amplitude {ampl}A.")
    return ampl


def sleep_and_record_comments(t, file_object, recording_name, cuttoff_time=0.01):
    """ Sleeps for t seconds, but records comments in the file represented by the file handler.
        If the user writes "quit()" or "exit()", the function imidiately returns 1, otherwise it returns 0."""
    # MAYBE ADD A SWITHC - if on windows just regular sleep, if on linux - comment recorder!
    start = time.perf_counter()
    print("Tracking started! Comment in here if needed:")
    print(f"Comments for {recording_name}:", file=file_object)
    while (t - (time.perf_counter() - start)) > cuttoff_time:
        # SELECT MAY BE AN ISSUE FOR WINDOWS!!!
        i, o, e = select([sys.stdin], [], [], max(cuttoff_time, t - (time.perf_counter() - start)))

        if i:
            line = sys.stdin.readline().strip()
            if line.lower() == "exit()" or line.lower() == "quit()":
                return 1
            else:
                print(line, file=file_object)

    print("Tracking period ended!")
    print("", file=file_object)

    # Just to be shure the program slept just enough! :)
    time.sleep(max(0.0, (t - (time.perf_counter() - start))))
    return 0


def print_prerun_checklist():
    """ Outputs to stdout a checklist of items that need to be checked pre run."""
    print(f"\n\n{'-'*10} PRE-RUN CHECKLIST {'-'*10}")
    print("\t[1] All auto settings OFF.")
    print("\t[2] Framerate and Gamma settings ON, all else OFF.")
    print("\t[3] Choose camera mode (0 - 1080x1920 vs. 1 - 600x960). Check hw_config file:")
    print(f"\t[4] Current x_pixel_count is {get_hw_config()['x_pixels']} (from hw_config file).")
    print("\t[5] Center rod.")
    print("\t[6] Set viewing field, recenter rod.")
    print("\t[8] See max fps, min fps, set appropriate (const.) shutter.")
    print(f"{'-'*17} END {'-'*17}\n\n")
