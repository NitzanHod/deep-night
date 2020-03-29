import json
import os
import time


def get_exposure_time(image_path):
    """ Returns the exposure time of an image from the given path """
    if not os.path.exists(image_path):
        raise FileNotFoundError()

    os.system('exiftool -json -ExposureTime ' + image_path + ' > exp_times.json')

    with open('exp_times.json') as f:
        data = json.load(f)

    os.remove('exp_times.json')

    for element in data:
        exp_time = element["ExposureTime"]

    if type(exp_time) is str and '/' in exp_time:
        num, den = exp_time.split('/')
        result = 0 + (float(num) / float(den))
        exp_time = result

    return exp_time


def get_aperture(image_path):
    """ Returns the exposure time of an image from the given path """
    if not os.path.exists(image_path):
        raise FileNotFoundError()

    os.system('exiftool.exe -json -Aperture ' + image_path + ' > apertures.json')

    with open('apertures.json') as f:
        data = json.load(f)

    os.remove('apertures.json')

    for element in data:
        return element["Aperture"]
