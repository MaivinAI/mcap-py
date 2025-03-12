## Installation
This Python app allows user to parse MCAP files and view each H264 frame using Matplotlib.  It also allows the user to overlay the bounding boxes from the detection topic (in blue) onto the frame. Furthermore, the app also has a feature to feed it an Onnx model which can inference the frame and overlay those bounding boxes (in red) onto the frame..

### Install Python Dependencies
The following libraries must be installed as part of the Ubuntu environment using the following command:
```bash
sudo apt-get install python3-tk
sudo apt-get install build-essential
sudo apt-get install libxcb-xinerama0
sudo apt-get install qtbase5-dev
```
Python library requirements must be installed via Pip:
```bash
pip install -r requirements.txt
``` 

### Run the Script
The script can be run simply with:
```bash
python viewer.py <mcap_file>
``` 
The MCAP file must have the following topics included:
* /camera/h264
* /camera/info
* /model/boxes2d

To remove the MCAP bounding boxes, run the command with the -b option:
```bash
python viewer.py -b <mcap_file>
``` 
To view the bounding boxes with an ONNX model, include the model with -m option:
```bash
python viewer.py -b -m <ONNX_model> <mcap_file>
```
Both sets of bounding boxes can be combined:
```bash
python viewer.py -m <ONNX_model> -b <mcap_file>
```

# License
This project is licensed under the AGPL-3.0 or under the terms of the DeepView AI Middleware Commercial License.

# Support
Commercial Support is provided by Au-Zone Technologies through the [DeepView Support](https://support.deepviewml.com) site.
