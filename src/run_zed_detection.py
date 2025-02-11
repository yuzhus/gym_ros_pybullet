#!/usr/bin/env python3

'''
    The object detection node for ZED camera
'''

import pyzed.sl as sl
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from threading import Lock, Thread
from time import sleep

# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import Point
from collections import defaultdict
import ogl_viewer.viewer as gl

track_history = defaultdict(lambda: [])
lock = Lock()
run_signal = False
exit_signal = False
old_object_id = 13
PerspectiveTransformResolution = (380,370)


def GetTransformInputs():
    pt_A = (int(442), int(174))
    pt_B = (int(372), int(542))
    pt_C = (int(751), int(539))
    pt_D = (int(721), int(179))

    width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                                [0, maxHeight - 1],
                                [maxWidth - 1, maxHeight - 1],
                                [maxWidth - 1, 0]])
    return pt_A, pt_B, pt_C, pt_D, input_pts, output_pts

def xywh2abcd(xywh):
    global center, output
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max

    center = [xywh[0],xywh[1]]

    return output, center

def detections_to_custom_box(boxes, track_cls):
    output = []
    for i, box in enumerate(boxes):
        xywh = box.numpy() # det is a tensor 
        # print('xywh is ', xywh)
        # https://www.stereolabs.com/docs/object-detection/custom-od/
        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        [obj.bounding_box_2d, center] = xywh2abcd(xywh)
        obj.label = track_cls[i]
        # print('obj.label = ', obj.label)
        # obj.object_id = det[0].boxes.id.int().cpu().tolist()
        # obj.probability = det[0].boxes.conf.cpu().numpy().astype(int)
        output.append(obj)
    return output

def torch_thread(weights, img_size, conf_thres=0.25, iou_thres=0.45):
    
    global boxes, track_ids, track_cls, annotated_frame, tansformed_frame, exit_signal, run_signal, output

    print("Intializing Network...")

    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            img = cv2.cvtColor(tansformed_frame, cv2.COLOR_BGRA2RGB) # maybe have influence on the different color???????????????????????
            # https://docs.ultralytics.com/modes/predict/#video-suffixes
            # det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[0].cpu().numpy().boxes
            results = model.track(img, persist=True, imgsz=img_size, conf=conf_thres, iou=iou_thres)
            annotated_frame = results[0].plot()
             # Get the boxes and track IDs
            if results[0].boxes.id != None:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().numpy().astype(int)
                track_cls = results[0].boxes.cls.cpu().numpy().astype(int)
                print('The cls are',track_cls)

            # ZED CustomBox format (with inverse letterboxing tf applied)
            output = detections_to_custom_box(boxes, track_cls)
            lock.release()
            run_signal = False

        sleep(0.01)

def ros_pybullet_publisher(x_goal,y_goal,z_goal,x_obs,y_obs,z_obs,size_x, size_y, size_z):
    global new_object   
    rospy.loginfo('Published a new object')
    pub_goal.publish(x_goal, y_goal, z_goal)
    pub_obs.publish(x_obs, y_obs, z_obs)
    pub_obs_size.publish(size_x,size_y,size_z)
    rate.sleep()

def main():
    global boxes, track_ids, track_cls, annotated_frame, tansformed_frame, exit_signal, run_signal, output, z_point0, z_point4
    
    pt_A, pt_B, pt_C, pt_D, input_pts, output_pts = GetTransformInputs()

    capture_thread = Thread(target=torch_thread, kwargs={'weights': weights, 'img_size': img_size, "conf_thres": conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    # Create a ZED camera object
    zed = sl.Camera()

    input_type = sl.InputType()
    # if if_svo is not None:
    #     input_type.set_from_svo_file(if_svo)

    # Set configuration parameters
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD720 # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 30 # Set fps at 30
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    # Open the camera
    err = zed.open(init_params)
    if (err != sl.ERROR_CODE.SUCCESS) :
        print(repr(err))
        zed.close()
        exit(-1)

    # Set object detection parameters
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    if obj_param.enable_tracking :
        # Set positional tracking parameters
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # Enable positional tracking (important to the gl depth viewer)
        zed.enable_positional_tracking(positional_tracking_parameters)
    zed.enable_object_detection(obj_param)
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution

    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_configuration.resolution

    # Declare your sl.Mat matrices
    image_zed_tmp = sl.Mat()
    image_zed = sl.Mat()
    display_resolution = sl.Resolution(PerspectiveTransformResolution[0] ,PerspectiveTransformResolution[1])
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_zed_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Camera pose
    cam_w_pose = sl.Pose()

    # GL viewer
    viewer = gl.GLViewer()
    gl_resolution = sl.Resolution(PerspectiveTransformResolution[0], PerspectiveTransformResolution[1]) 
    viewer.init(zed.get_camera_information().camera_model, gl_resolution, obj_param.enable_tracking)
    point_cloud = sl.Mat(gl_resolution.width, gl_resolution.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    key = ' '

    while key != 27 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            ## Retrieve image ##############################
            lock.acquire()
            zed.retrieve_image(image_zed_tmp, sl.VIEW.LEFT, sl.MEM.CPU, image_size)

            # To recover data from sl.Mat to use it with opencv, use the get_data() method
            # It returns a numpy array that can be used as a matrix with opencv
            image_ocv = image_zed_tmp.get_data()
            frame = cv2.resize(image_ocv, (1280, 720))
            
            # 4 red markers
            cv2.circle(frame, pt_A, 2, (0,0,255), -1)
            cv2.circle(frame, pt_B, 2, (0,0,255), -1)
            cv2.circle(frame, pt_C, 2, (0,0,255), -1)
            cv2.circle(frame, pt_D, 2, (0,0,255), -1)

            # Apply Geometrical Transformation
            matrix = cv2.getPerspectiveTransform(input_pts, output_pts)
            tansformed_frame = cv2.warpPerspective(frame, matrix, (PerspectiveTransformResolution[0] ,PerspectiveTransformResolution[1]))

            cv2.imshow("ZED LEFT VIEW", frame)
            # cv2.imshow("Transformed Bird's Eye View", tansformed_frame)
            
            lock.release()
            run_signal = True
            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            lock.acquire()
            zed.ingest_custom_box_objects(output) # Very important for getting 3D bounding box
            lock.release()

            ## Retrieve object ##############################
            zed.retrieve_objects(objects, obj_runtime_param)

            viewer.updateData(point_cloud, objects)
            gl_viewer_available = viewer.is_available()

            for i in range(len(objects.object_list)):
                bounding_3Dbox = np.array(objects.object_list[i].bounding_box)
            # print('3d bounding box is', bounding_3Dbox)

            ## Retrieve image ##############################
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            frame_zed = image_zed.get_data()
            resized_frame_zed = cv2.resize(frame_zed, (1280, 720))
            tansformed_frame_zed= cv2.warpPerspective(resized_frame_zed, matrix, (PerspectiveTransformResolution[0] ,PerspectiveTransformResolution[1]))

            # 2D rendering
            np.copyto(image_zed_ocv, tansformed_frame_zed)
            # Blue color in BGR 
            color = (255, 0, 0) 
            # Line thickness of 2 px 
            thickness = 2
            # if len(detections) != 0:
            obstacle_height = 0
            z_point0 = 0
            z_point4 = 0
            box_size_x = 0.12
            box_size_y = 0.12
            box_size_z = 0.12
            if len(boxes) != 0:
                for i, cls in enumerate(track_cls):
                    if cls == 0: # obstacle
                        x_obs = boxes.numpy().astype(int)[i][0] # xywh
                        y_obs = boxes.numpy().astype(int)[i][1] # xywh
                        for j in range(len(bounding_3Dbox)):
                            if j == 0:
                                z_point0 = bounding_3Dbox[j][2]
                                # print(z_point0)
                            if j == 4:
                                z_point4 = bounding_3Dbox[j][2]
                                # print(z_point4)
                        obstacle_height = abs(z_point0 - z_point4)
                        print('the height of the object is ', obstacle_height)
                        if obstacle_height > 0.41:
                            box_size_z = 0.24
                        else:
                            box_size_z = 0.12

                    else: # goal
                        x_goal = boxes.numpy().astype(int)[i][0]
                        y_goal = boxes.numpy().astype(int)[i][1]
                cv2.imshow("Transformed Bird's Eye View with YOLOv8", annotated_frame)
               
                ########################################################
                ## talk to the pybullet gym environment 
                dx = 0.1 # mapping parameters
                dy = 0.12

                xx_obs = -0.3 - (x_obs/PerspectiveTransformResolution[0])*0.482 - dx
                yy_obs = (y_obs/PerspectiveTransformResolution[1])*0.587 + dy
                zz_obs = 0.06

                xx_goal = -0.3 - (x_goal/PerspectiveTransformResolution[0])*0.482 - dx
                yy_goal = (y_goal/PerspectiveTransformResolution[1])*0.587 + dy
                zz_goal = 0.01

                ros_pybullet_publisher(xx_goal,yy_goal,zz_goal,xx_obs,yy_obs,zz_obs,box_size_x,box_size_y,box_size_z)
                ########################################################

            key = cv2.waitKey(10)
            if key == 27:
                exit_signal = True
        else:
            exit_signal = True

    exit_signal = True
    cv2.destroyAllWindows()
    viewer.exit()
    zed.disable_object_detection()
    zed.close()
    print("\nFINISH")

if __name__ == '__main__':


    rospy.init_node('zed_detection_node', anonymous=True)

    # Set the logging system
    rospack = rospkg.RosPack()
    pkg_path = rospack.get_path('gym_ros_pybullet')
    weights_dir = pkg_path + '/src/Yolov8/best_digitaltwin.pt'
    # env = wrappers.Monitor(env, outdir, force=True) 
    rospy.loginfo ( "ZED detection started")

    # Loads parameters from the ROS param server
    # Parameters are stored in a yaml file inside the config directory
    # They are loaded at runtime by the launch file
    img_size = rospy.get_param("/img_size")
    conf_thres = rospy.get_param("/conf_thres")
    weights = weights_dir


    # rospy.init_node('ros_pybullet_talker', anonymous=True)
    pub_goal = rospy.Publisher('/digital_twin_goal_position', Point, queue_size=10)
    pub_obs = rospy.Publisher('/digital_twin_obstacle_position', Point, queue_size=10)
    pub_obs_size = rospy.Publisher('/obstacle_size', Point, queue_size=10)
    rate = rospy.Rate(50) # 1hz

    with torch.no_grad():
        main()





