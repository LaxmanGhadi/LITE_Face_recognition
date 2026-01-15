import RPi.GPIO as GPIO
import subprocess
import time
import os
import sys
import signal
from GoogleSheet import mark_attendance
from Start import current_name
BUTTON_PIN = 17
SCRIPT_TO_RUN = "/home/omotecrpi/Downloads/LITE_Face_recognition-main/Face_recog_v4/Start.py"
# Function to check if target script exists
if not os.path.exists(SCRIPT_TO_RUN):
    print(f"ERROR: {SCRIPT_TO_RUN} not found")
    sys.exit(1)
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
# Track process to avoid multiple launches 
process = None
def start_script(channel):
    global process
    if process is None or process.poll() is not None:
        print("Button pressed! Starting script...")
        process = subprocess.Popen(["/usr/bin/python3", SCRIPT_TO_RUN])
    else:
        print("Script already running, ignoring button press.")
GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=start_script, bouncetime=1000)


STOP_BUTTON  = 27
GPIO.setup(STOP_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
def stop_script(channel):    
    global process
    if process and process.poll() is None:
        print("Stopping script...")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process = None
    else:
        print("Script not running")
GPIO.add_event_detect(STOP_BUTTON, GPIO.FALLING, stop_script, bouncetime=500)

STOP_BUTTON  = 26
GPIO.setup(STOP_BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)
def attendance_button_pressed(channel):
    # Here `current_name` should come from your face recognition module
    try:
        mark_attendance(current_name)
        print(f"Attendance marked for {current_name}")
    except NameError:
        print("No recognized name yet!")

# Main loop
try:
    print("Listener started, waiting for button presses...")
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    GPIO.cleanup()
    if process is not None:
        process.terminate()
