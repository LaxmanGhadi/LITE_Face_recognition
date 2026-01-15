from display_text import Start_message, disp_txt
import cv2
import Face_recog
import signal
import sys
import time

running = True
cap = None
current_name = None
# ---------------- CLEAN SHUTDOWN ----------------
def shutdown(signum=None, frame=None):
    global running, cap
    print("Shutting camera down...")

    running = False

    if cap is not None:
        cap.release()

    disp_txt("Turning off camera", 3)
    time.sleep(0.5)
    disp_txt("Camera Off", 2)

    # Optional: clear screen (depends on your display lib)
    disp_txt("", 0.1)

    sys.exit(0)

# Catch stop signals (systemd / kill / button stop)
signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

# ---------------- STARTUP ----------------
Start_message()
disp_txt("Turning on camera", 3)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    disp_txt("Camera Error", 3)
    sys.exit(1)

person = "Unknown"

# ---------------- MAIN LOOP ----------------
while running:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = Face_recog.detect_face(rgb_img)

    if face is None:
        disp_txt("No person", 0.1)
    else:
        score, name = Face_recog.check_person(face)
        if score * 100 > 70:
            disp_txt(name, 0.1)
            current_name = name
        else:
            disp_txt("Unknown person", 0.1)

# If loop exits naturally
shutdown()
