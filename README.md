## Installation

The python app allows user to parse MCAP files; View each H264 frame using OpenCV and overlay the bounding boxes on to the frame. Furthermore, the app also has a feature to feed it an Onnx model which can inference the frame
### Install Python Dependencies

```bash
pip install -r requirements.txt
``` 
### Run the Script
```bash
python mcap_parser.py -f <path_to_mcap_file>
``` 
