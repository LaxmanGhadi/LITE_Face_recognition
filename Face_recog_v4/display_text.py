from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306  # SSD1306 driver works with SSD1315-class controllers
from PIL import Image, ImageDraw, ImageFont
import time

# Setup I2C (address 0x3C is standard)
serial = i2c(port=1, address=0x3C)
device = ssd1306(serial)

# Create blank image
width = device.width
height = device.height
image = Image.new("1", (width, height))
draw = ImageDraw.Draw(image)

draw.rectangle((0, 0, width, height), outline=0, fill=0)
font = ImageFont.load_default()

def disp_txt(given_text, s_time):
    draw.text((0, 0), given_text, font=font, fill=255)
    device.display(image)
    time.sleep(s_time)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)

def Start_message():
    draw.text((0, 0), "Starting System.", font=font, fill=255)
    device.display(image)
    time.sleep(0.5)

    draw.text((0, 0), "Starting System..", font=font, fill=255)
    device.display(image)
    time.sleep(0.5)

    draw.text((0, 0), "Starting System...", font=font, fill=255)
    device.display(image)
    time.sleep(0.5)

    draw.text((0, 0), "Starting System....", font=font, fill=255)
    device.display(image)
    time.sleep(0.5)
    draw.rectangle((0, 0, width, height), outline=0, fill=0)

