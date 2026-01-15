import os
import subprocess
import sys

# ===== CONFIGURE THESE =====
SERVICE_NAME = "always-active-script"
USER = "omotecrpi"
PYTHON_EXEC = "/usr/bin/python3"
TARGET_SCRIPT = "/home/omotecrpi/Downloads/LITE_Face_recognition-main/Face_recog_v4/listner.py"
# ==========================

SERVICE_FILE = f"/etc/systemd/system/{SERVICE_NAME}.service"

SERVICE_CONTENT = f"""[Unit]
Description=Always Active Python Script
After=multi-user.target

[Service]
ExecStart={PYTHON_EXEC} {TARGET_SCRIPT}
WorkingDirectory={os.path.dirname(TARGET_SCRIPT)}
Restart=always
User={USER}

[Install]
WantedBy=multi-user.target
"""

def run(cmd):
    print(f"> {cmd}")
    subprocess.check_call(cmd, shell=True)

def main():
    if os.geteuid() != 0:
        print("❌ This script MUST be run with sudo")
        sys.exit(1)

    print("✅ Creating systemd service...")

    with open(SERVICE_FILE, "w") as f:
        f.write(SERVICE_CONTENT)

    run("systemctl daemon-reload")
    run(f"systemctl enable {SERVICE_NAME}")
    run(f"systemctl start {SERVICE_NAME}")

    print("\n🎉 DONE!")
    print(f"Service '{SERVICE_NAME}' is now active and will start on every boot.")
    print(f"Check status with: systemctl status {SERVICE_NAME}")

if __name__ == "__main__":
    main()
