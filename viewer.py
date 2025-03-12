import argparse  
import io
import av  
import numpy as np  
import cv2  
import logging  
import matplotlib.pyplot as plt
from mcap.summary import Summary
from mcap.reader import make_reader
from edgefirst.schemas.sensor_msgs import CameraInfo as Info  # Custom message module for camera information  
from edgefirst.schemas.foxglove_msgs import ImageAnnotations as Boxes  # Custom message module for image annotations
from edgefirst.schemas.foxglove_msgs import CompressedVideo  # Custom message module for H264 decoding 
from edgefirst.schemas.edgefirst_msgs import Detect # Custom message module for detection

# Initialize logger
logging.basicConfig(level=logging.INFO)  # Set up logging configuration
logger = logging.getLogger(__name__)  # Create a logger object
av.logging.set_level(5)

# Initialize an input/output buffer for storing raw data
rawData = io.BytesIO()
# Open the AV container to parse H.264 video format
container = av.open(rawData, format="h264", mode='r' )

# Default frame dimensions
frame_height = 1080
frame_width = 1920

# Set key press used in show boxes loop
KEY_PRESSED = None

# Function to extract an image frame from H264 and keep track of key and I frames
def get_image(message, frame_position):

    rawData.write(message)  # Write message data to the buffer
    rawData.seek(frame_position)  # Move the buffer position to the specified frame position
    mcap_image = None  # Initialize variable to store the image

    have_key_frame=False
    for packet in container.demux():
        if packet.size == 0:  # Skip empty packets
            continue
        frame_position += packet.size  # Update frame position

        if not have_key_frame:
            if not packet.is_keyframe:
                continue
            else:
                have_key_frame = True

        try:
            for frame in packet.decode():  # Decode frames from the packet
                # Convert the frame to RGB format and store it
                mcap_image = cv2.cvtColor(frame.to_ndarray(format='rgb24'), cv2.COLOR_BGR2RGB)
        except Exception as e:  
            logger.warning(f"Unable to decode frame: {e}")
            continue  
    return mcap_image  # Return the decoded image

# Function to draw bounding boxes on the image by inferencing the model
def draw_model_boxes(dets, boxes, classes, mcap_image, thickness):
    try:
        # Iterate over detected objects
        for i in range(dets[0]):
            # Calculate coordinates of the bounding box
            start_point = (round(boxes[i][0] * mcap_image.shape[1]), round(boxes[i][1] * mcap_image.shape[0]))
            end_point = (round(boxes[i][2] * mcap_image.shape[1]), round(boxes[i][3] * mcap_image.shape[0]))
            # Draw the bounding box with a color based on the class
            if classes[0][i] == 0:
                mcap_image = cv2.rectangle(mcap_image, start_point, end_point, (0,0,255), thickness)  # Red color
            elif classes[0][i] == 1:
                mcap_image = cv2.rectangle(mcap_image, start_point, end_point, (0,255,0), thickness)  # Green color
            elif classes[0][i] == 2:
                mcap_image = cv2.rectangle(mcap_image, start_point, end_point, (255,255,0), thickness)  # Yellow color
    except Exception as e:  
        logger.error("Error inferencing model to draw bounding boxes: %s", e)  
    return mcap_image  # Return the image with bounding boxes drawn

# Function to run the object detection model and draw bounding boxes
def run_model(mcap_image, model, thickness):
    try:
        # Import required modules, importing them here as it takes time and are not always required
        from tensorflow.image import combined_non_max_suppression
        import onnxruntime
        
        # Preprocess the input image for the model
        input_img = cv2.resize(mcap_image, (640, 640))
        input_img = np.transpose(input_img, [2,0,1])
        input_img = np.expand_dims(input_img, 0).astype(np.float32)
        input_img = input_img / 255
        
        # Load the model and perform inference
        session = onnxruntime.InferenceSession(str(model), providers=['CPUExecutionProvider'])
        outputs = session.run(['classes', 'boxes'], {'images': input_img})
        
        # Perform non-maximum suppression to filter bounding boxes
        boxes, scores, classes, dets = combined_non_max_suppression(outputs[0].reshape([1,25200,1,4]) / 640, outputs[1], 100, 100, score_threshold=0.35)
        boxes = boxes.numpy()[0]
        scores = scores.numpy()
        classes = classes.numpy()
        dets = dets.numpy()
        
        # Draw bounding boxes on the image
        mcap_image = draw_model_boxes(dets, boxes, classes, mcap_image, thickness)
    except Exception as e:  
        logger.error("Error running model: %s", e)  
    return mcap_image  # Return the image with bounding boxes drawn

def press(event):
    global KEY_PRESSED
    KEY_PRESSED = event.key

# Function to display the image
def show_image(frame_id, mcap_image):
    try:
        rgb_image = cv2.cvtColor(mcap_image, cv2.COLOR_BGR2RGB) # Fix color representation
        plt.imshow(rgb_image)  # Display the corrected image using matplotlib
        plt.gcf().canvas.mpl_connect('key_press_event', press)
        plt.axis('off')  # Turn off axis
        plt.suptitle(f"Frame {frame_id}")  # Set title
        plt.title("Press 'q' to stop or any other key to continue")
        logger.info(f"Showing Frame {frame_id}")
        plt.draw()  # Show the image
    except KeyboardInterrupt:
        logger.info("Keyboard Interrupt Detected.  Exiting...")
        exit()
    except Exception as e:
        logger.error("Error displaying image:", e)
        return

    while(True):
        if plt.waitforbuttonpress(0):
            plt.close()
            break
            
    global KEY_PRESSED
    if KEY_PRESSED.lower() == 'q':
        logger.info("Quitting MCAP Viewer")
        exit()
    KEY_PRESSED = None
    return
    
# Function to set the image size based on camera information
def set_image_size(message, scale):
    global frame_height  # Access the global variable
    frame_height = int(Info.deserialize(message.data).height*scale)  # Update the frame height
    global frame_width  # Access the global variable
    frame_width = int(Info.deserialize(message.data).width*scale)  # Update the frame width
    return False  # Return False to indicate that scale is set

# Function to get the closest time to sync frame and boxes
def get_closest_time(boxes_map, frame_time):
    min_difference = float('inf')  # Initialize to positive infinity
    closest_time = None
    for time_key in boxes_map.keys():
        difference = abs(frame_time - time_key)
        if difference < min_difference:
            min_difference = difference
            closest_time = time_key
    return closest_time

# Function to draw the custom boxes 
def draw_custom_bbox(message, boxes_map, frame_time, mcap_image, scale, display_bbox,thickness):
    try:
        boxes = Detect.deserialize(message.data)
        box_time = boxes.header.stamp.sec + (boxes.header.stamp.nanosec / 1e9) # Get the box time
        boxes_map[box_time] = boxes.boxes
        closest_time = get_closest_time(boxes_map, frame_time)
        if mcap_image is not None:
            logger.info(f"Found {len(boxes_map[closest_time])} boxes at time {closest_time}")
        for points in boxes_map[closest_time]:  # Iterate over annotation points
            if points and mcap_image is not None:  # Check if points and image are available
                x = int((points.center_x - points.width / 2) * frame_width/scale)
                y = int((points.center_y - points.height / 2) * frame_height/scale)
                w = int(points.width * frame_width/scale)
                h = int(points.height * frame_height/scale)
                if display_bbox:
                    cv2.rectangle(mcap_image, (x, y), (x + w, y + h), (255, 0, 0), thickness) # Draw a bounding box on the image
    except:
        logger.warning("Error in deserializing bounding boxes, just showing Image")


def is_topic_present(summary:Summary, topic:str) -> bool:
    for id, channel in summary.channels.items():
        if channel.topic == topic:
            logger.info(f"Found topic {topic}")
            return True
    logger.error(f"Did not find topic {topic}")
    return False   

# Function to visualize the MCAP file
def visualizer(mcap_file, model, scale, thickness, display_bbox, scale_not_set):
    frame_position, frame_id = 0, 0  # Initialize frame position and ID
    mcap_image = None  # Initialize image variable
    frame_time = 0 # Stores the time when the frame was received to sync with the boxes
    boxes_map = {} # Creates a hash of the boxes to match with frame time 
    boxes_topic = "/model/boxes2d"
    try:
        with open(mcap_file, "rb") as f:  # Open the MCAP file for reading
            logger.info(f"Opening {mcap_file}")
            reader = make_reader(f)  # Create a reader object for reading messages
            sum = reader.get_summary()
            for topic in ["/camera/info", "/camera/h264", "/model/boxes2d" ]:
                if not is_topic_present(sum, topic):
                    logger.error(f"Cannot view {mcap_file} without topic {topic}.  Exiting")
                    exit()

            for schema, channel, message in reader.iter_messages():  # Iterate over messages in the file
                if channel.topic == "/camera/info" and scale_not_set:  # Check if camera info and scale are not set
                    scale_not_set = set_image_size(message, scale)  # Set the image size based on camera info
                    
                if channel.topic == "/camera/h264":  # Check if the topic is camera H.264
                    frame_id = frame_id + 1  # Increment frame ID
                    image_data =  CompressedVideo.deserialize(message.data) # Deserialize the message data to get H264 frames
                    frame_time = image_data.timestamp.sec + (image_data.timestamp.nanosec / 1e9) # Get the frame time
                    mcap_image = get_image(bytes(image_data.data), frame_position)  # Get the image frame from the message
                
                if channel.topic == boxes_topic:  # Check if the topic is 2D bounding boxes
                    draw_custom_bbox(message, boxes_map, frame_time, mcap_image, scale, display_bbox, thickness)
                        
                    if mcap_image is not None:  # Check if image is available
                        mcap_image = cv2.resize(mcap_image, (frame_width, frame_height))  # Resize the image
                        if model:  # Check if a model is provided
                            mcap_image = run_model(mcap_image, model, thickness)  # Run object detection model
                        show_image(frame_id, mcap_image)  # Show the image

    except Exception as e:  
        logger.error("Error in visualizer: %s", e)  

# Main function to parse command-line arguments and start visualization
def main():
    
    parser = argparse.ArgumentParser(description='Process MCAP to view images with bounding boxes.')  # Create an argument parser
    parser.add_argument('-m', '--model', nargs='?', const=True, default=False, help='Run the MCAP frames through a custom ONNX model to display bounding box.')  # Add model argument
    parser.add_argument('mcap_file', type=str, help='MCAP that needs to be parsed') # Add MCAP file argument
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='Resizing factor to view the final image 0.1-1.0. Default: 1.0')  # Add scale argument
    parser.add_argument('-t', '--thickness', type=int, default=2, help='Choose the thickness of the bounding box. Default: 2')  # Add thickness argument
    parser.add_argument('-b', '--display_bbox', action='store_false', help='Choose to view the bounding box. Default: True') # Gives an option to display the Bounding Boxes
    opt = parser.parse_args()  # Parse command-line arguments

    scale_not_set = True  # Flag to check if scale is initially set
    try:
        visualizer(opt.mcap_file, opt.model, opt.scale, opt.thickness, opt.display_bbox, scale_not_set)  # Visualize the MCAP file
    except Exception as e:  
        logger.error("Unable to parse the user inputs: %s", e)  

if __name__ == '__main__':
    main()  
