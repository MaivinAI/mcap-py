
import argparse  
import io
import av  
import numpy as np  
import cv2  
import logging  
from mcap.reader import make_reader
from edgefirst.schemas.sensor_msgs import CameraInfo as Info  # Custom message module for camera information  
from edgefirst.schemas.foxglove_msgs import ImageAnnotations as Boxes  # Custom message module for image annotations
from edgefirst.schemas.foxglove_msgs import CompressedVideo  # Custom message module for H264 decoding 
from edgefirst.schemas.edgefirst_msgs import Detect # Custom message module for detection

# Initialize logger
logging.basicConfig(level=logging.INFO)  # Set up logging configuration
logger = logging.getLogger(__name__)  # Create a logger object

# Initialize an input/output buffer for storing raw data
rawData = io.BytesIO()
# Open the AV container to parse H.264 video format
container = av.open(rawData, format="h264", mode='r')

# Default frame dimensions
frame_height = 1080
frame_width = 1920

# Function to extract an image frame from H264 and keep track of key and I frames
def get_image(message, frame_position):
    rawData.write(message)  # Write message data to the buffer
    rawData.seek(frame_position)  # Move the buffer position to the specified frame position
    mcap_image = None  # Initialize variable to store the image
    
    # Iterate over packets in the container to decode frames
    for packet in container.demux():
        try:
            if packet.size == 0:  # Skip empty packets
                continue
            frame_position += packet.size  # Update frame position
            for frame in packet.decode():  # Decode frames from the packet
                # Convert the frame to RGB format and store it
                mcap_image = cv2.cvtColor(frame.to_ndarray(format='rgb24'), cv2.COLOR_BGR2RGB)
        except Exception as e:  
            logger.warning("Ubable to decode frame: %s", e)
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

# Function to display the image
def show_image(frame_id, mcap_image, key):
    try:
        # Display the image
        if key is not None and key == 27:  # Check if ESC key is pressed
            exit()  # Exit the program if ESC key is pressed
        cv2.imshow("frame " + str(frame_id), mcap_image)  # Show the image
        key = cv2.waitKey(0)  # Wait for a key press
        cv2.destroyAllWindows()  # Close all OpenCV windows
    except Exception as e:  
        logger.error("Error displaying image: %s", e)  
    return key  # Return the key pressed by the user

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
        for points in boxes_map[closest_time]:  # Iterate over annotation points
            if points and mcap_image is not None:  # Check if points and image are available
                x = int((points.center_x - points.width / 2) * frame_width/scale)
                y = int((points.center_y - points.height / 2) * frame_height/scale)
                w = int(points.width * frame_width/scale)
                h = int(points.height * frame_height/scale)
                if display_bbox:
                    cv2.rectangle(mcap_image, (x, y), (x + w, y + h), (255, 0, 0), thickness) # Draw a bounding box on the image
    except:
        logger.warning("Error in deserializing custom boxes, just showing Image")

# Function to draw foxglove boxes
def draw_foxglove_bbox(message, boxes_map, frame_time, mcap_image, display_bbox, thickness):
    try:
        boxes = Boxes.deserialize(message.data)  # Deserialize the message data to get bounding boxes
        box_time = boxes.points[0].timestamp.sec + (boxes.points[0].timestamp.nanosec / 1e9) # Get the box time
        boxes_map[box_time] = boxes.points
        closest_time = get_closest_time(boxes_map, frame_time)
        for points_annotation in boxes_map[closest_time]:  # Iterate over annotation points
            points = points_annotation.points  # Get the points of the bounding box
            if points and mcap_image is not None:  # Check if points and image are available
                box_points = [(int(point.x), int(point.y)) for point in points]  # Convert points to integers
                min_x = min(point[0] for point in box_points)  # Get minimum X coordinate
                min_y = min(point[1] for point in box_points)  # Get minimum Y coordinate
                max_x = max(point[0] for point in box_points)  # Get maximum X coordinate
                max_y = max(point[1] for point in box_points)  # Get maximum Y coordinate
                if display_bbox:
                    cv2.rectangle(mcap_image, (min_x, min_y), (max_x, max_y), (255, 0, 0), thickness) # Draw a bounding box on the image
    except:
        logger.warn("Error in deserializing foxglove boxes, just showing Image")

# Function to visualize the MCAP file
def visualizer(mcap_file, model, scale, thickness, display_bbox, custom, scale_not_set):
    frame_position, frame_id = 0, 0  # Initialize frame position and ID
    key, mcap_image = None, None  # Initialize key and image variables
    frame_time = 0 # Stores the time when the frame was recived to sync with the boxes
    boxes_map = {} # Creates a hash of the boxes to match with frame time 
    try:
        with open(mcap_file, "rb") as f:  # Open the MCAP file for reading
            reader = make_reader(f)  # Create a reader object for reading messages
            for schema, channel, message in reader.iter_messages():  # Iterate over messages in the file
                if channel.topic == "/camera/info" and scale_not_set:  # Check if camera info and scale are not set
                    scale_not_set = set_image_size(message, scale)  # Set the image size based on camera info
                
                if channel.topic == "/camera/h264":  # Check if the topic is camera H.264
                    frame_id = frame_id + 1  # Increment frame ID
                    image_data =  CompressedVideo.deserialize(message.data) # Deserialize the message data to get H264 frames
                    frame_time = image_data.timestamp.sec + (image_data.timestamp.nanosec / 1e9) # Get the frame time
                    mcap_image = get_image(bytes(image_data.data), frame_position)  # Get the image frame from the message
                    
                if channel.topic == "/detect/boxes2d":  # Check if the topic is 2D bounding boxes
                    if custom:
                        draw_custom_bbox(message, boxes_map, frame_time, mcap_image, scale, display_bbox, thickness)
                    else: 
                        draw_foxglove_bbox(message, boxes_map, frame_time, mcap_image, display_bbox, thickness)
                    if mcap_image is not None:  # Check if image is available
                        mcap_image = cv2.resize(mcap_image, (frame_width, frame_height))  # Resize the image
                        if model:  # Check if a model is provided
                            mcap_image = run_model(mcap_image, model, thickness)  # Run object detection model
                        key = show_image(frame_id, mcap_image, key)  # Show the image and get the key pressed by the user
    except Exception as e:  
        logger.error("Error in visualizer: %s", e)  

# Main function to parse command-line arguments and start visualization
def main():
    
    parser = argparse.ArgumentParser(description='Process MCAP to view images with bounding boxes.')  # Create an argument parser
    parser.add_argument('-m', '--model', nargs='?', const=True, default=False, help='Run the frame through a custom model to display bounding box. Specify the model name after --model. Default: False')  # Add model argument
    parser.add_argument('mcap_file', type=str, help='MCAP that needs to be parsed') # Add MCAP file argument
    parser.add_argument('-s', '--scale', type=float, default=1.0, help='Resizing factor to view the final image 0.1-1.0. Default: 1.0')  # Add scale argument
    parser.add_argument('-t', '--thickness', type=int, default=2, help='Choose the thickness of the bounding box. Default: 2')  # Add thickness argument
    parser.add_argument('-b', '--display_bbox', action='store_true', help='Choose to view the bounding box. Default: False') # Gives an option to display the Bounding Boxes
    parser.add_argument('-c', '--custom', action='store_true', help='Choose to view the kind of bounding box [Custom Boxes, Foxglove Boxes]. Default: False') # Allows user swtitch between custom and foxglove schema
    opt = parser.parse_args()  # Parse command-line arguments

    scale_not_set = True  # Flag to check if scale is initially set
    try:
        visualizer(opt.mcap_file, opt.model, opt.scale, opt.thickness, opt.display_bbox, opt.custom, scale_not_set)  # Visualize the MCAP file
    except Exception as e:  
        logger.error("Unable to parse the user inputs: %s", e)  

if __name__ == '__main__':
    main()  
