[![pipeline status](https://gitlab.motius.de/diamond-finder/cv-pipeline/badges/dev/pipeline.svg)](https://gitlab.motius.de/diamond-finder/cv-pipeline/commits/dev)
[![coverage report](https://gitlab.motius.de/diamond-finder/cv-pipeline/badges/dev/coverage.svg)](https://gitlab.motius.de/diamond-finder/cv-pipeline/commits/dev)

# Hilti QA Computer Vision Pipeline

## Installation

Create virtualenv and install requirements
```bash
cd /path/to/project
python3 -m venv .cv-diamond

# activate for windows
source .cv-diamond/Scripts/activate

# activate for unix based systems
source .cv-diamond/bin/activate

# get camera-module
git submodule init && git submodule update
pip install -e . && pip install -e camera-module/
```
If you are **not** running this on Windows, pip install of pypylon will
fail, so please comment out pypylon in the requirements.txt and build it
from source as stated in 
[the Basler pypylon repository](https://github.com/basler/pypylon).

## Requirements
Please see [requirements.txt](./requirements.txt) for more info.

## Usage
For testing connect camera and set boolearn use_camera to True **or**
set the path to your images in config.py 

Finally, run:
```bash
python scripts/cv-pipeline.py
```
Make sure virtual environment is activated, when running the cv-pipeline

## Tools
#### Show Template for Segment Detection
For debugging purposes you can show the compressed template using
```bash
python scripts/show_template.py
```
It will show the template in a OpenCV window.

# Data Workshop
## Fixed bounding boxes and polygons
There is a new script to run the fixed bounding boxes setup. The
 installation is the same as before no new requirements.
 
To run the script:
```bash
# 1. activate virtual environment
source .cv-diamond/bin/activate

# 2. update path to files in fixed-bb-pipeline.py

# 3. run new python script for fixed bounding boxes
python scripts/fixed-bb-pipeline.py

# when you're done you can deactivate the virtual environment by
deactivate
```

Currently there are three setups which you can change by setting the
 booleans on top in [fixed-bb-pipeline.py](./scripts/fixed-bb-pipeline.py)
1. Fixed bounding boxes no border (`use_polygon=False, border=False`)
2. Fixed bounding boxes with border (`use_polygon=False, border=True`)
3. Polygons fit to the segment shape (`use_polygon=True, border=False`)

If you want to see the processing for each segment set 
`self.process_segments(seg, verbose=True)` (verbose to True) in
 [image2.py](./src/hilti/image2.py).# billion_diamonds
