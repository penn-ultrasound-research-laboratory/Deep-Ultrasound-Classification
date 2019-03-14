import json
import os

with open ("../V2.0_Processed/manifest.json") as f:
    manifest = json.load(f)

# Searching for grayscale images that are listed in the manifest but the files do not exist
for k in manifest:
    for frame in manifest[k]:
        if frame["IMAGE_TYPE"] == "GRAYSCALE" and not os.path.isfile("{0}/{1}/{2}".format(frame["TUMOR_TYPE"], k, frame["FRAME"])):
            print("{0}: {1}".format(k, frame["FRAME"]))
