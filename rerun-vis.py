import math
import cv2
import os
import argparse
import io
import av
import numpy as np
import rerun as rr
import logging
import tqdm
import sys
import zstandard
from mcap.reader import make_reader
# Custom message module for camera information
from edgefirst.schemas.sensor_msgs import CameraInfo as Info
from edgefirst.schemas.sensor_msgs import PointCloud2, PointField, PointFieldDatatype
# Custom message module for H264 decoding
from edgefirst.schemas.foxglove_msgs import CompressedVideo
# Custom message module for detection
from edgefirst.schemas.edgefirst_msgs import Mask
import struct
import traceback
import io
# Initialize logger
logging.basicConfig(level=logging.INFO)  # Set up logging configuration
logger = logging.getLogger(__name__)  # Create a logger object

rawData = None
container = None
zstd_decomp = zstandard.ZstdDecompressor()


def init_h264():
    global rawData
    global container
    # Initialize an input/output buffer for storing raw h264 data
    try:
        os.remove("tmp.h264")
    except OSError:
        pass
    rawData = io.open("tmp.h264", "a+b")
    # Open the AV container to parse H.264 video format

    # Block errors/warnings resulting from lack of keyframes
    av.logging.set_level(av.logging.PANIC)
    container = av.open("tmp.h264", format="h264", mode='r')


def uninit_h264():
    rawData.close()
    try:
        os.remove("tmp.h264")
    except OSError:
        pass


def get_image(message, frame_position):
    """
    Extracts an image frame from H264 and keep track of key and I frames
    """
    rawData.write(message)  # Write message data to the buffer
    # Move the buffer position to the specified frame position
    rawData.seek(frame_position)
    mcap_image = None  # Initialize variable to store the image

    # Iterate over packets in the container to decode frames
    for packet in container.demux():
        try:
            if packet.size == 0:  # Skip empty packets
                continue
            # if not packet.is_keyframe:
            #     continue
            frame_position += packet.size  # Update frame position

            for frame in packet.decode():  # Decode frames from the packet
                # Convert the frame to RGB format and store it
                # mcap_image = cv2.cvtColor(frame.to_ndarray(
                #     format='rgb24'), cv2.COLOR_BGR2RGB)
                mcap_image = frame

        except Exception as e:
            if "Errno 1094995529" not in str(e):
                logger.warning("Unable to decode frame: %s", e)
            else:
                # error due to not starting with a keyframe
                pass
            continue
    return mcap_image  # Return the decoded image


# Camera matrix that produces bounding boxes in 1920x1080 pixel coordinates
# cam_mtx = np.array([
#     [1260, 0, 960,],
#     [0, 1260, 540,],
#     [0, 0, 1,],
# ])

# Camera matrix that produces normalized bounding boxes
cam_mtx = np.array([
    [1260/1920, 0, 960/1920,],
    [0, 1260/1080, 540/1080,],
    [0, 0, 1,],
])

# Converts lzxydwh (label, z, x, y, depth, width, height) coulmn vector into (x_min, y_min, z_min) column vector
zxydwh2xyzmin = np.array([
    [0, 0, 1, 0, 0, -0.5, 0],
    [0, 0, 0, 1, 0, 0, -0.5],
    [0, 1, 0, 0, 0, 0, 0],
])

# Converts lzxydwh (label, z, x, y, depth, width, height) coulmn vector into (x_max, y_max, z_max) column vector
zxydwh2xyzmax = np.array([
    [0, 0, 1, 0, 0, 0.5, 0],
    [0, 0, 0, 1, 0, 0, 0.5],
    [0, 1, 0, 0, 0, 0, 0],
])

# Converts (x, y, 1) column vector from camera coordinate system to the image coordinate system.
coord_cnvt = np.array([
    [-1, 0, 1],
    [0, -1, 1],
    [0, 0, 1],
])


def convert3d2d(bb3d: np.ndarray, width=1920, height=1080):
    """
    Projects 3D bounding boxes from real world coordinates into image coordinates

    Keyword arguments:
    bb3d -- The n x 7 bounding box from the annotation file. Format is (label, z, x, y, depth, width, height) 
    width -- the width of the output image (default 1920)
    height -- the height of the output image (default 1080)

    Example:
    ```
    convert3d2d(np.array([[0, 9.641414, -0.612911, 1.2554842, 0.70922685, 0.7092271, 1.8080416],
                          [0, 9.641414, -0.612911, 1.2554842, 0.70922685, 0.7092271, 1.8080416]]))
    ```
    """
    xyzmin = zxydwh2xyzmin @ bb3d.transpose()
    bbmin = cam_mtx @ xyzmin
    bbmin /= bbmin[2, :]
    bbmin = coord_cnvt @ bbmin
    bbmin = bbmin[:2, :]
    bbmin[0, :] *= width
    bbmin[1, :] *= height

    xyzmax = zxydwh2xyzmax @ bb3d.transpose()
    bbmax = cam_mtx @ xyzmax
    bbmax /= bbmax[2, :]
    bbmax = coord_cnvt @ bbmax
    bbmax = bbmax[:2, :]
    bbmax[0, :] *= width
    bbmax[1, :] *= height

    bbmax_ = np.maximum(bbmax, bbmin)
    bbmin = np.minimum(bbmax, bbmin)
    sizes = bbmax_-bbmin
    return bbmin.transpose(), sizes.transpose()


# https://stackoverflow.com/a/12729229
def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # return wa*Ia + wb*Ib + wc*Ic + wd*Id
    return (Ia.T*wa).T + (Ib.T*wb).T + (Ic.T*wc).T + (Id.T*wd).T


def get_classes(mask, points, size):
    """
    Projects the points onto the mask and classifies them based on the argmax of that projected position in the mask
    """
    width = mask.shape[1]
    height = mask.shape[0]
    classes = []
    for i in range(len(points)):
        xy = points[i]
        x = xy[0] / width
        y = xy[1] / height
        if not (0 < x < 1):
            classes.append(0)
            continue
        if not (0 < y < 1):
            classes.append(0)
            continue
        ind = 0
        vals = bilinear_interpolate(
            mask, x * width, y * height)
        if np.max(vals) > 1:
            ind = np.argmax(vals)
        for ang in range(0, 360, 45):
            dx = math.sin(math.radians(ang)) * size[i][0]
            dy = math.cos(math.radians(ang)) * size[i][0]
            if ind == 0:
                vals = bilinear_interpolate(
                    mask, x * width + dx, y * height + dy)
                if np.max(vals) > 1:
                    ind = np.argmax(vals)
        classes.append(ind)
    return np.asarray(classes)


def filter_occlusions(points, center_2d, size_2d, classes):
    """
    This reclassifies the non-background points that are more than 1m behind another non-background point and overlap their widths into background points
    """
    combined = np.concatenate(
        (center_2d, size_2d, np.expand_dims(classes, 1)), axis=1)

    distance = [math.sqrt(points[p][1]**2 + points[p][2] ** 2)
                for p in range(points.shape[0])]

    # 2d.x 2d.y 2d.width 2d.height class distance index
    index = np.expand_dims(np.asarray(
        [x for x in range(combined.shape[0])]), 1)
    combined = np.concatenate(
        (combined, np.expand_dims(distance, 1), index), axis=1)
    combined = combined.tolist()
    # sort by distance
    combined.sort(key=lambda x: x[5])
    to_clear = []
    for i in range(len(combined)):
        if combined[i][4] == 0:
            continue
        for j in range(i):
            if combined[j][4] == 0:
                continue
            if abs(combined[j][0] - combined[i][0]) < combined[j][2] + combined[i][2] and abs(combined[j][5] - combined[i][5]) > 1.0:
                to_clear.append(int(combined[i][6]))
    return to_clear


class Points3D():
    x: float
    y: float
    z: float
    class_: int
    fields: dict

    def __init__(self):
        self.fields = dict()


def make_blueprint():
    import rerun.blueprint as rrb
    my_blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="/3d/mask"),
                rrb.Spatial2DView(origin="/3d/video")
            ),
            rrb.Spatial3DView(origin="/")
        ),
        collapse_panels=True
    )
    return my_blueprint


def visualizer(mcap_file, image_scaling, rerun_file):
    frame_position, frame_id = 0, 0  # Initialize frame position and ID
    mcap_image = None  # Initialize image variables
    class_colors = [
        (128, 128, 128, 0),  # Background
        (255, 0, 0),  # Person
        (0, 255, 0),  # Vehicle
        (0, 0, 255),  # Pavement
    ]

    # Default frame dimensions
    frame_height = 1080
    frame_width = 1920

    frame_height = int(frame_height*image_scaling)
    frame_width = int(frame_width*image_scaling)
    try:
        with open(mcap_file, "rb") as f:  # Open the MCAP file for reading
            # Create a reader object for reading messages

            rr.init("Raivin MCAP Visualizer", spawn=False)
            # save in a file
            rr.save(rerun_file, default_blueprint=make_blueprint())
            # rr.spawn(default_blueprint=make_blueprint()) # view live

            reader = make_reader(f)
            last_mask = None
            rr.log(
                "3d/video/mask",  # Applies to all entities below "masks".
                rr.AnnotationContext(
                    [
                        rr.AnnotationInfo(
                            id=0, label="Background", color=class_colors[0]),
                        rr.AnnotationInfo(
                            id=1, label="Person", color=class_colors[1]),
                        rr.AnnotationInfo(
                            id=2, label="Vehicle", color=class_colors[2]),
                        rr.AnnotationInfo(
                            id=3, label="Pavement", color=class_colors[3])
                    ],
                ),
                static=True,
            )
            count = 0
            speeds = []
            for schema, channel, message in reader.iter_messages():
                count += 1
                if channel.topic == "/radar/targets":
                    # Default code for decoding /radar/targets
                    radar_data = PointCloud2.deserialize(message.data)
                    endian_format = ">" if radar_data.is_bigendian else "<"
                    for i in range(radar_data.height):
                        for j in range(radar_data.width):
                            point_start = \
                                (i*radar_data.width+j) * radar_data.point_step
                            for f in radar_data.fields:
                                f: PointField = f
                                val = 0
                                if f.datatype == PointFieldDatatype.FLOAT32.value:
                                    arr = bytearray(
                                        radar_data.data[(point_start + f.offset):(point_start + f.offset + 4)])
                                    val = struct.unpack(
                                        f'{endian_format}f', arr)[0]
                                elif f.datatype == PointFieldDatatype.FLOAT64.value:
                                    arr = bytearray(
                                        radar_data.data[(point_start + f.offset):(point_start + f.offset + 4)])
                                    val = struct.unpack(
                                        f'{endian_format}f', arr)[0]
                                else:
                                    logger.warning(
                                        "Found non float xyz data in points field. Integer parsing not supported yet")
                                if f.name == 'speed':
                                    speeds.append(val)

            # Determine normalization coefficents for speed
            max_speed = max(speeds)
            min_speed = min(speeds)
            if abs(max_speed) > abs(min_speed):
                speed_color_mult = 255 / (abs(max_speed))
            else:
                speed_color_mult = 255 / (abs(min_speed))

            for schema, channel, message in tqdm.tqdm(reader.iter_messages(), total=count):
                if channel.topic == "/camera/h264":  # Check if the topic is camera H.264
                    frame_id = frame_id + 1  # Increment frame ID
                    # Deserialize the message data to get H264 frames
                    image_data = CompressedVideo.deserialize(message.data)
                    frame_time = image_data.timestamp.sec + \
                        (image_data.timestamp.nanosec / 1e9)  # Get the frame time
                    # Get the image frame from the message
                    mcap_image = get_image(
                        bytes(image_data.data), frame_position)

                    if mcap_image:
                        image = mcap_image.to_ndarray(format="bgr24")
                        image = cv2.resize(image, dsize=(
                            frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                        res, image = cv2.imencode(".jpg", image)
                        rr.log(
                            "3d/video", rr.ImageEncoded(contents=bytes(image), format=rr.ImageFormat.JPEG))
                if channel.topic == "/radar/targets":
                    radar_data = PointCloud2.deserialize(message.data)
                    radar_points = []
                    endian_format = ">" if radar_data.is_bigendian else "<"
                    for i in range(radar_data.height):
                        for j in range(radar_data.width):
                            radar_point = Points3D()
                            point_start = \
                                (i*radar_data.width+j) * radar_data.point_step
                            for f in radar_data.fields:
                                f: PointField = f

                                val = 0
                                if f.datatype == PointFieldDatatype.FLOAT32.value:
                                    arr = bytearray(
                                        radar_data.data[(point_start + f.offset):(point_start + f.offset + 4)])
                                    val = struct.unpack(
                                        f'{endian_format}f', arr)[0]
                                elif f.datatype == PointFieldDatatype.FLOAT64.value:
                                    arr = bytearray(
                                        radar_data.data[(point_start + f.offset):(point_start + f.offset + 4)])
                                    val = struct.unpack(
                                        f'{endian_format}f', arr)[0]
                                else:
                                    logger.warning(
                                        "Found non float xyz data in points field. Integer parsing not supported yet")
                                if f.name == 'x':
                                    radar_point.x = val
                                elif f.name == 'y':
                                    radar_point.y = val
                                elif f.name == 'z':
                                    radar_point.z = val
                                else:
                                    radar_point.fields[f.name] = val
                            radar_points.append(radar_point)

                    #  (label, z, x, y, depth, width, height)
                    POINT_RADIUS = 0.25
                    points = [[0, p.x, p.y, p.z, POINT_RADIUS*2, POINT_RADIUS*2, POINT_RADIUS*2,]
                              for p in radar_points]
                    points = np.asarray(points)

                    mins_2d, size_2d = convert3d2d(
                        points, width=frame_width, height=frame_height)
                    centers_2d = mins_2d + size_2d/2
                    colors = [0, 0, 0, 0]
                    if last_mask is not None:
                        # convert the 2D coordinates to Mask dimensions
                        mask_height = last_mask.shape[0]
                        mask_width = last_mask.shape[1]
                        centers_2d[:, 0] *= mask_width/frame_width
                        size_2d[:, 0] *= mask_width/frame_width
                        centers_2d[:, 1] *= mask_height/frame_height
                        size_2d[:, 1] *= mask_height/frame_height
                        classes = get_classes(last_mask, centers_2d, size_2d)
                        filter = filter_occlusions(
                            points, centers_2d, size_2d, classes)

                        for i in filter:
                            classes[i] = 0
                        colors = [class_colors[c]
                                  for c in classes]

                        # convert the 2D coordinates to back to Image dimensions
                        centers_2d[:, 0] /= mask_width/frame_width
                        size_2d[:, 0] /= mask_width/frame_width
                        centers_2d[:, 1] /= mask_height/frame_height
                        size_2d[:, 1] /= mask_height/frame_height

                    # Determine colour by speed field.
                    colors = []
                    for p in radar_points:
                        if p.fields["speed"] < 0:
                            colors.append((255 - (int((p.fields["speed"]) * -speed_color_mult)), 
                                           255 - (int((p.fields["speed"]) * -speed_color_mult)),
                                           255))
                        else:
                            colors.append((255, 
                                           255 - (int((p.fields["speed"]) * speed_color_mult)), 
                                           255 - (int((p.fields["speed"]) * speed_color_mult))))
                    rr.log("3d/video/points",
                           rr.Points2D(positions=centers_2d, radii=size_2d[:, 0]/2, colors=colors))
                    rr.log("3d/radar", rr.Points3D(
                        positions=[[p.x, p.y, p.z] for p in radar_points], radii=POINT_RADIUS, colors=colors))

                if channel.topic == "/detect/mask":
                    msg = Mask.deserialize(message.data)
                    mask = msg.mask
                    if msg.encoding == "zstd":
                        mask = zstd_decomp.decompress(bytes(mask))
                        mask = [x for x in mask]
                    elif msg.encoding == "":
                        pass
                    else:
                        logger.error(
                            f"Unknown encoding type {msg.encoding} in mask")
                    mask = np.asarray(mask, dtype=np.uint8)
                    mask = mask.reshape((msg.height, msg.width, -1))
                    last_mask = mask
                    rr.log("3d/masktensor", rr.Image(mask[:, :, 1:]))
                    mask = cv2.resize(mask, (frame_width, frame_height),
                                      interpolation=cv2.INTER_LINEAR)
                    mask = np.argmax(mask, axis=2)
                    rr.log("3d/video/mask", rr.SegmentationImage(mask))
                    rr.log("3d/mask", rr.SegmentationImage(mask))

    except Exception as e:
        tb = io.StringIO()
        traceback.print_tb(e.__traceback__, 10, tb)
        tb.seek(0)
        logger.error(f"Error in visualizer: {e}\nTraceback:\n{tb.read()}")

        # Main function to parse command-line arguments and start visualization


def main():

    parser = argparse.ArgumentParser(
        description='Process MCAP to view images with bounding boxes.')  # Create an argument parser
    parser.add_argument('-m', '--model', nargs='?', const=True, default=False,
                        help='Run the frame through a custom model to display bounding box. Specify the model name after --model. Default: False')  # Add model argument
    parser.add_argument('-s', '--scale', type=float, default=.5,
                        help='Resizing factor to view the final image 0.1-1.0. Default: 1.0')  # Add scale argument
    parser.add_argument('mcap_file', type=str,
                        help='MCAP that needs to be parsed')
    parser.add_argument('-o', '--output_file', type=str,
                        default="data.rrd", help='The name of the output file')
    opt = parser.parse_args()  # Parse command-line arguments
    init_h264()
    visualizer(opt.mcap_file, opt.scale, opt.output_file)
    uninit_h264()


if __name__ == '__main__':
    main()
