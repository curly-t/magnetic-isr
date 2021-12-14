import time, socket, serial, re
from warnings import warn
from datetime import datetime
from os import path, mkdir
from .config import get_config
from functools import cache


@cache
def get_hw_config():
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

    return {"serial_port": serial_port, "logfilename": logfilename, "bd": baudrate, "x_pixels": x_pixels,
            "server_port": server_port, "server_address": server_address, "max_current": max_current}


def setup_serial():
    hw_config = get_hw_config()
    return serial.Serial(port=hw_config["serial_port"], timeout=1, baudrate=hw_config["bd"])


def send_serial_command(s, cmd):
    logfilename = get_hw_config()["logfilename"]
    with open(logfilename, "r") as logfile:
        print(f"SENDING TO SERIAL: {cmd}", file=logfile)
        s.write(cmd)
        time.sleep(0.1)
        line = s.readline()
        print(f"READING FROM SERIAL: {line}", file=logfile)


def get_calibrated_current_value(channel, currentA):
    if channel == 1:
        return 0.5224 * currentA + 0.001412
    elif channel == 2:
        return 0.4427 * currentA + 0.001495
    else:
        return 0.0


def cmd_stop(s):
    send_serial_command(s, b'STOP\r\n')


def cmd_start(s):
    send_serial_command(s, b'START\r\n')


def cmd_set_frequency(s, frequency):
    send_serial_command(s, b'FREQ=%.3f\r\n' % frequency)


def cmd_set_current(s, offsetA, amplitudeA):
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
    send_serial_command(s, b'OFFLED=%.3f\r\n' % offsetA)
    send_serial_command(s, b'AMPLED=%.3f\r\n' % amplitudeA)


def send_to_server(message):
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Get the address and port where the server is listening
    hw_conf = get_hw_config()
    server_address = (hw_conf["server_address"], hw_conf["server_port"])

    with open(hw_conf["logfilename"], "r") as logfile:
        try:
            # Connect to server
            sock.connect(server_address)

            # Send data
            print(f"SENDING TO SERVER: {message}", file=logfile)
            sock.send(message.encode())

            # Look for the response
            data = sock.recv(1024)
            print(f"RECEIVED FROM SERVER {data}", file=logfile)

        except Exception as e:
            print(f"Exception {e.__class__} OCCURED!", file=logfile)
            warn(f"Exception {e.__class__} OCCURED!")

    # Close the socket - WHY WAS IT COMMENTED OUT? MAYBE THIS CAN CAUSE PROBLEMS - I WILL TRY WITH IT ENABLED AND SEE
    # PAZI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    sock.close()


def enable_led(led_offset=0.09, led_ampl=0.0):
    with setup_serial() as s:
        time.sleep(0.1)
        cmd_set_led_current(s, led_offset, led_ampl)
        cmd_set_frequency(s, 0.1)
        time.sleep(0.1)
        cmd_start(s)


def disable_all():
    with setup_serial() as s:
        time.sleep(0.1)
        cmd_set_led_current(s, 0., 0.)
        cmd_stop(s)


def disable_all_coils():
    with setup_serial() as s:
        time.sleep(0.1)
        cmd_stop(s)


def make_measurement_run_folder():
    dt = datetime.now()
    measurement_folder = get_config()["meas"]
    date_folder_path = path.join(measurement_folder, dt.strftime('%m-%d-%Y'))

    if not path.isdir(date_folder_path):
        mkdir(date_folder_path)

    print("You are starting a new run! How would you like to name the folder?")
    while True:
        folder_name = str(input())

        if not path.isdir(path.join(date_folder_path, folder_name)):
            break
        print("Folder name already exists! Choose a new one!")

    full_folder_path = path.join(date_folder_path, folder_name)
    mkdir(full_folder_path)

    return full_folder_path


def check_current(ampl, offs):
    max_curr = get_hw_config()["max_current"]
    assert (ampl + offs) <= max_curr
    assert (ampl - offs) > 0.


def check_frequency(freq, max_time):
    assert (max_time * freq) > 4
