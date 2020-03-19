import os
import glob
from PIL import Image

path = r"C:\Users\as70\Pictures\Vessel-IEEE\Results"
for file in glob.glob(os.path.join(path, "*.tif")):
    print(file)
    folder, base = os.path.split(file)
    name = os.path.splitext(base)[0]
    outfile = os.path.join(folder, name +".png")
    try:
        im = Image.open(file)
        print("Generating jpeg for %s" % name)
        im.thumbnail(im.size)
        im.save(outfile, "PNG", quality=100)
    except Exception as e:
        print(e)