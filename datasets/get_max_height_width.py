from PIL import Image
import os
import sys

path = sys.argv[1]
# set an initial value which no image will meet
maxw = -1
maxh = -1

for subdir, dirs, files in os.walk(path):
    for image in files:
        # get the image height & width
        image_location = os.path.join(subdir, image)
        im = Image.open(image_location)
        data = im.size
        # if the width is higher than the last image, we have a new "winner"
        w = data[0]
        if w > maxw:
            newmaxw = w, image_location
            maxw = w
        # if the height is higher than the last image, we have a new "winner"
        h = data[1]
        if h > maxh:
            newmaxh = h, image_location
            maxh = h
# finally, print the values and corresponding files
print("maxwidth", newmaxw)
print("maxheight", newmaxh)
