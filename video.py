from yolo.config import ConfigParser
from yolo.utils.box import visualize_boxes
import matplotlib.pyplot as plt
import argparse
import os
import cv2
argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/predict_coco.json",
    help='config file')

argparser.add_argument(
    '-i',
    '--image',
    default="tests/samples/sample.jpeg",
    help='path to image file')

# Load params
execution_path = os.getcwd()
input_file_path=os.path.join(execution_path, "videoplayback.mp4")
output_file_path=os.path.join(execution_path, "video11111-detected")
frames_per_second=30
minimum_percentage_probability=40, 
log_progress=True
__model_type = "yolov3"

# input_file_path="", 
camera_input=None
# output_file_path="", 
# frames_per_second=20,
frame_detection_interval=1
# minimum_percentage_probability=50, 
# log_progress=False,
display_percentage_probability=True
display_object_name=True
save_detected_video=True
per_frame_function=None
per_second_function=None
per_minute_function=None
video_complete_function=None
return_detected_frame=False
detection_timeout=None


# 1. create yolo model & load weights
args = argparser.parse_args()
config_parser = ConfigParser(args.config)
model = config_parser.create_model(skip_detect_layer=False)
detector = config_parser.create_detector(model)
# ///////////////////////////////////////////////////////////////////



output_frames_dict = {}
output_frames_count_dict = {}

input_video = cv2.VideoCapture(input_file_path)
output_video_filepath = output_file_path + '.avi'

frame_width = int(input_video.get(3))
frame_height = int(input_video.get(4))
output_video = cv2.VideoWriter(output_video_filepath, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                frames_per_second,
                                (frame_width, frame_height))

counting = 0
predicted_numbers = None
scores = None
detections = None
detection_timeout_count = 0
video_frames_count = 0

if(__model_type == "yolov3"):

    while (input_video.isOpened()):
        ret, frame = input_video.read()

        if (ret == True):

            detected_frame = frame.copy()

            video_frames_count += 1
            if (detection_timeout != None):
                if ((video_frames_count % frames_per_second) == 0):
                    detection_timeout_count += 1

                if (detection_timeout_count >= detection_timeout):
                    break

            output_objects_array = []

            counting += 1

            if (log_progress == True):
                print("Processing Frame : ", str(counting))

            check_frame_interval = counting % frame_detection_interval

            if (counting == 1 or check_frame_interval == 0):
                try:
                    # detected_frame, output_objects_array = self.__detector.detectObjectsFromImage(
                    #     input_image=frame, input_type="array", output_type="array",
                    #     minimum_percentage_probability=minimum_percentage_probability,
                    #     display_percentage_probability=display_percentage_probability,
                    #     display_object_name=display_object_name)
                    detected_frame_pre, output_objects_array, pred = detector.detect(frame, 0.2)
                    visualize_boxes(frame, detected_frame_pre, output_objects_array, pred, config_parser.get_labels())
                    detected_frame = frame

                    # plt.imshow(detected_frame)
                    # plt.show()
                except:
                    print('none')
                    None


            output_frames_dict[counting] = output_objects_array

            output_objects_count = {}
            for eachItem in output_objects_array:
                eachItemName = eachItem
                try:
                    output_objects_count[eachItemName] = output_objects_count[eachItemName] + 1
                except:
                    output_objects_count[eachItemName] = 1

            output_frames_count_dict[counting] = output_objects_count

            if (save_detected_video == True):
                output_video.write(detected_frame)

            if (counting == 1 or check_frame_interval == 0):
                if (per_frame_function != None):
                    if (return_detected_frame == True):
                        per_frame_function(counting, output_objects_array, output_objects_count,
                                            detected_frame)
                    elif (return_detected_frame == False):
                        per_frame_function(
                            counting, output_objects_array, output_objects_count)

            if (per_second_function != None):
                if (counting != 1 and (counting % frames_per_second) == 0):

                    this_second_output_object_array = []
                    this_second_counting_array = []
                    this_second_counting = {}

                    for aa in range(counting):
                        if (aa >= (counting - frames_per_second)):
                            this_second_output_object_array.append(
                                output_frames_dict[aa + 1])
                            this_second_counting_array.append(
                                output_frames_count_dict[aa + 1])

                    for eachCountingDict in this_second_counting_array:
                        for eachItem in eachCountingDict:
                            try:
                                this_second_counting[eachItem] = this_second_counting[eachItem] + \
                                    eachCountingDict[eachItem]
                            except:
                                this_second_counting[eachItem] = eachCountingDict[eachItem]

                    for eachCountingItem in this_second_counting:
                        this_second_counting[eachCountingItem] = int(
                            this_second_counting[eachCountingItem] / frames_per_second)

                    if (return_detected_frame == True):
                        per_second_function(int(counting / frames_per_second),
                                            this_second_output_object_array, this_second_counting_array,
                                            this_second_counting, detected_frame)

                    elif (return_detected_frame == False):
                        per_second_function(int(counting / frames_per_second),
                                            this_second_output_object_array, this_second_counting_array,
                                            this_second_counting)

            if (per_minute_function != None):

                if (counting != 1 and (counting % (frames_per_second * 60)) == 0):

                    this_minute_output_object_array = []
                    this_minute_counting_array = []
                    this_minute_counting = {}

                    for aa in range(counting):
                        if (aa >= (counting - (frames_per_second * 60))):
                            this_minute_output_object_array.append(
                                output_frames_dict[aa + 1])
                            this_minute_counting_array.append(
                                output_frames_count_dict[aa + 1])

                    for eachCountingDict in this_minute_counting_array:
                        for eachItem in eachCountingDict:
                            try:
                                this_minute_counting[eachItem] = this_minute_counting[eachItem] + \
                                    eachCountingDict[eachItem]
                            except:
                                this_minute_counting[eachItem] = eachCountingDict[eachItem]

                    for eachCountingItem in this_minute_counting:
                        this_minute_counting[eachCountingItem] = int(
                            this_minute_counting[eachCountingItem] / (frames_per_second * 60))

                    if (return_detected_frame == True):
                        per_minute_function(int(counting / (frames_per_second * 60)),
                                            this_minute_output_object_array, this_minute_counting_array,
                                            this_minute_counting, detected_frame)

                    elif (return_detected_frame == False):
                        per_minute_function(int(counting / (frames_per_second * 60)),
                                            this_minute_output_object_array, this_minute_counting_array,
                                            this_minute_counting)

        else:
            break

    if (video_complete_function != None):

        this_video_output_object_array = []
        this_video_counting_array = []
        this_video_counting = {}

        for aa in range(counting):
            this_video_output_object_array.append(
                output_frames_dict[aa + 1])
            this_video_counting_array.append(
                output_frames_count_dict[aa + 1])

        for eachCountingDict in this_video_counting_array:
            for eachItem in eachCountingDict:
                try:
                    this_video_counting[eachItem] = this_video_counting[eachItem] + \
                        eachCountingDict[eachItem]
                except:
                    this_video_counting[eachItem] = eachCountingDict[eachItem]

        for eachCountingItem in this_video_counting:
            this_video_counting[eachCountingItem] = this_video_counting[
                eachCountingItem] / counting

        video_complete_function(this_video_output_object_array, this_video_counting_array,
                                this_video_counting)

    input_video.release()
    output_video.release()
