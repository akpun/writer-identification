from PIL import Image
import os
import sys

path = sys.argv[1]
# set an initial value which no image will meet
minw = 10000000
minh = 10000000

for subdir, dirs, files in os.walk(path):
    for image in files:
        # get the image height & width
        image_location = os.path.join(subdir, image)
        im = Image.open(image_location)
        data = im.size
        # if the width is lower than the last image, we have a new "winner"
        w = data[0]
        if w < minw:
            newminw = w, image_location
            minw = w
        # if the height is lower than the last image, we have a new "winner"
        h = data[1]
        if h < minh:
            newminh = h, image_location
            minh = h
# finally, print the values and corresponding files
print("minwidth", newminw)
print("minheight", newminh)
