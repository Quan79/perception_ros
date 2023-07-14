#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
from cv_bridge import CvBridge

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from ros_yolov5_msgs.msg import BoundingBox, BoundingBoxes
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped



class Yolo_Dect:
    def __init__(self):

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '')

        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param(
            '~image_topic', '/camera_front/color/image_raw')
        depth_topic = rospy.get_param(
            '~depth_topic', '/camera_front/depth/image_rect_raw')  
        depth_camera_info_topic = rospy.get_param(
            '~depth_camera_info_topic', '/camera_front/depth/camera_info'
        )      
        color_camera_info_topic = rospy.get_param(
            '~color_camera_info_topic', '/camera_front/color/camera_info'
        )    
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        conf = rospy.get_param('~conf', '0.55')

        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom',
                                    path=weight_path, source='local')

        # which device will be used
        # use cpu only
        if (rospy.get_param('/use_cpu', 'true')):
            self.model.cpu()
        # else:
        #     self.model.cuda()

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.camera_info_depth = CameraInfo()
        self.camera_info_color = CameraInfo()
        self.getImageStatus = False

        #initialize cv bridge
        self.bridge = CvBridge()
 
        # Load class color
        self.classes_colors = {}

        # image subscribe
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1)
                                          
        # depth subscribe
        self.depth_sub = rospy.Subscriber(depth_topic, Image, self.depth_callback,
                                          queue_size=1)
        
        #camera_info subscribe
        self.camera_info_color_sub = rospy.Subscriber(color_camera_info_topic, CameraInfo, self.camera_info_color_callback)
        
        #camera_info subscribe
        self.camera_info_depth_sub = rospy.Subscriber(depth_camera_info_topic, CameraInfo, self.camera_info_depth_callback)
        
        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)

        self.cones_pub = rospy.Publisher(
            "/cones_position",  Path, queue_size=1)

        self.image_pub = rospy.Publisher(
            '/yolov5/detection_image',  Image, queue_size=1)
        
        self.depth_pub = rospy.Publisher(
            '/yolov5/depth_image',  Image, queue_size=1)
        
        # if no image messages
        while (not self.getImageStatus):
            rospy.loginfo("waiting for image.")
            rospy.sleep(2)


    def image_callback(self, color_image_msg):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = color_image_msg.header
        self.boundingBoxes.image_header = color_image_msg.header
        self.getImageStatus = True
        
        # self.color_image = np.frombuffer(color_image_msg.data, dtype=np.uint8).reshape(
        #     color_image_msg.height, color_image_msg.width, -1)
        self.color_image = self.bridge.imgmsg_to_cv2(color_image_msg, desired_encoding="bgr8")
        # color_image = color_image[..., ::-1].transpose((0, 3, 1, 2))
        # self.color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        
        copied_img = self.color_image.copy()
        
        results = self.model(copied_img)
        # xmin    ymin    xmax   ymax  confidence  class    name

        boxs = results.pandas().xyxy[0].values
        self.dectshow(copied_img, boxs, copied_img.shape[0], copied_img.shape[1])

        # cv2.waitKey(3)

    def depth_callback(self, depth_image_msg):
        # self.depth_image = np.frombuffer(depth_image_msg, dtype=np.uint16).reshape(depth_image_msg.height, depth_image_msg.width)
        depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, desired_encoding="passthrough")
                # 检查是否已经接收到彩色图像和深度图像
        # if self.color_image is not None and self.depth_image is not None:
        #     # 对齐和校准深度图像和彩色图像
        #     # depth_image_np = np.array(self.depth_image, dtype=np.float32)
        aligned_depth_image = self.align_depth_to_color(depth_image, self.camera_info_depth, self.camera_info_color)
        self.depth_image = aligned_depth_image
        
    def camera_info_depth_callback(self, camera_info_depth_msg):
        self.camera_info_depth = camera_info_depth_msg

    def camera_info_color_callback(self, camera_info_color_msg):
        # 保存相机的camera_info消息
        self.camera_info_color = camera_info_color_msg
        
        

    def align_depth_to_color(self, depth_image, camera_info_depth, camera_info_color):
        # 获取相机内参
        K_depth = np.array(camera_info_depth.K).reshape(3, 3)
        K_color = np.array(camera_info_color.K).reshape(3, 3)
        
        D_depth = np.array(camera_info_depth.D)
        D_color = np.array(camera_info_color.D)
        
        R_depth = np.array(camera_info_depth.R).reshape(3, 3)
        R_color = np.array(camera_info_color.R).reshape(3, 3)
        
        P_depth = np.array(camera_info_depth.P).reshape(3, 4)
        P_color = np.array(camera_info_color.P).reshape(3, 4)


        # 计算对齐映射
        map_x, map_y = cv2.initUndistortRectifyMap(K_depth, D_depth, R_depth, K_color, depth_image.shape[::-1], cv2.CV_32FC1)
        aligned_depth_image = cv2.remap(depth_image, map_x, map_y, cv2.INTER_LINEAR)
        
        return aligned_depth_image
            
    def dectshow(self, org_img, boxs, height, width):
        img = org_img.copy()

        count = 0
        for i in boxs:
            count += 1
        
        cones_msg = Path()


        for box in boxs:
            boundingBox = BoundingBox()
            boundingBox.probability =np.float64(box[4])
            boundingBox.xmin = np.int64(box[0])
            boundingBox.ymin = np.int64(box[1])
            boundingBox.xmax = np.int64(box[2])
            boundingBox.ymax = np.int64(box[3])
            boundingBox.num = np.int16(count)
            boundingBox.Class = box[-1]
            boundingBox.distance = np.float64('nan')

            if box[-1] in self.classes_colors.keys():
                color = self.classes_colors[box[-1]]
            else:
                color = np.random.randint(0, 183, 3)
                self.classes_colors[box[-1]] = color

            cv2.rectangle(img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (int(color[0]),int(color[1]), int(color[2])), 2)

            if box[1] < 20:
                text_pos_y = box[1] + 30
            else:
                text_pos_y = box[1] - 10
                
            cv2.putText(img, box[-1],
                        (int(box[0]), int(text_pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            
            center_x= int((box[0] + box[2]) / 2)
            center_y = int((box[1] + box[3]) / 2)
            self.distance = self.depth_image[center_y, center_x]

            boundingBox.distance = float ((self.distance)/1000)
            boundingBox.center_x = center_x
            boundingBox.center_y = center_y
            
            pose_stamped = PoseStamped()
            cones_msg.header.stamp = rospy.Time.now()
            cones_msg.header.frame_id = "vehicle_frame"
            pose_stamped.pose.position.z = self.distance /1000
            pose_stamped.pose.position.x = (center_x -638.79 )/645.7 * pose_stamped.pose.position.z
            pose_stamped.pose.position.y = (center_y -352.38 )/645.7 * pose_stamped.pose.position.z
            cones_msg.poses.append(pose_stamped)


            self.boundingBoxes.bounding_boxes.append(boundingBox)
            self.position_pub.publish(self.boundingBoxes)
        
        self.cones_pub.publish(cones_msg)
        self.publish_image(img, height, width)

        self.aligned_ros_depth_image = self.bridge.cv2_to_imgmsg(self.depth_image, encoding="passthrough")
        
        self.depth_pub.publish(self.aligned_ros_depth_image)
        # cv2.imshow('YOLOv5', img)

    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)
        # rospy.loginfo("Center point distance: {}".format(self.distance))

def main():

    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    rospy.spin()


if __name__ == "__main__":

    main()
