#!/usr/bin/env pipenv-shebang
# -*- coding:utf-8 -*-

# Copyright (c) 2023 SoftBank Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#!/usr/bin/env python3
'''Live demo of ClearGrasp
Will predict depth for all transparent objects on images streaming from a realsense camera using our API.
'''

import rospy
import argparse
import glob
import os
import shutil
import sys
import time

import cv2
import h5py
import numpy as np
import numpy.ma as ma
import termcolor
import yaml
from attrdict import AttrDict
# from PIL import Image
from sensor_msgs.msg import Image, CameraInfo
import message_filters
import rospkg
import cv_bridge

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from api import utils, depth_completion_api

class DepthCompletion(object):
    def __init__(self):

        pkg = rospkg.RosPack().get_path('cleargrasp')
        config_path = rospy.get_param('~config_path', pkg +'/config/config.yaml')
        with open(config_path) as fd:
            config_yaml = yaml.safe_load(fd)
        self.__config = AttrDict(config_yaml)
        self.__camera_info = rospy.wait_for_message('~color_camera_info', CameraInfo)
        self.__config.depth2depth.fx = self.__camera_info.K[0]
        self.__config.depth2depth.fy = self.__camera_info.K[4]
        self.__config.depth2depth.cx = self.__camera_info.K[2]
        self.__config.depth2depth.cy = self.__camera_info.K[5]

        self.__pub = rospy.Publisher('~output', Image, queue_size=10)

        self.__bridge = cv_bridge.CvBridge()
        rospy.loginfo('Loading Depth Completion API')

        self.__depthcomplete = depth_completion_api.DepthToDepthCompletion(normalsWeightsFile=self.__config.normals.pathWeightsFile,
                                                                    outlinesWeightsFile=self.__config.outlines.pathWeightsFile,
                                                                    masksWeightsFile=self.__config.masks.pathWeightsFile,
                                                                    normalsModel=self.__config.normals.model,
                                                                    outlinesModel=self.__config.outlines.model,
                                                                    masksModel=self.__config.masks.model,
                                                                    depth2depthExecutable=self.__config.depth2depth.pathExecutable,
                                                                    outputImgHeight=int(self.__config.depth2depth.yres),
                                                                    outputImgWidth=int(self.__config.depth2depth.xres),
                                                                    fx=int(self.__config.depth2depth.fx),
                                                                    fy=int(self.__config.depth2depth.fy),
                                                                    cx=int(self.__config.depth2depth.cx),
                                                                    cy=int(self.__config.depth2depth.cy),
                                                                    filter_d=self.__config.outputDepthFilter.d,
                                                                    filter_sigmaColor=self.__config.outputDepthFilter.sigmaColor,
                                                                    filter_sigmaSpace=self.__config.outputDepthFilter.sigmaSpace,
                                                                    maskinferenceHeight=self.__config.masks.inferenceHeight,
                                                                    maskinferenceWidth=self.__config.masks.inferenceWidth,
                                                                    normalsInferenceHeight=self.__config.normals.inferenceHeight,
                                                                    normalsInferenceWidth=self.__config.normals.inferenceWidth,
                                                                    outlinesInferenceHeight=self.__config.normals.inferenceHeight,
                                                                    outlinesInferenceWidth=self.__config.normals.inferenceWidth,
                                                                    min_depth=self.__config.depthVisualization.minDepth,
                                                                    max_depth=self.__config.depthVisualization.maxDepth,
                                                                    tmp_dir=pkg+'/tmp')

        color_sub = message_filters.Subscriber('~color', Image, queue_size=10)
        depth_sub = message_filters.Subscriber('~depth', Image, queue_size=10)
        self.__sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 10, 0.1)
        self.__sync.registerCallback(self.__callback)
        rospy.loginfo('Depth Completion API loaded')

    def __callback(self, color_msg, depth_msg):
        rospy.loginfo('cb')
        color_img = self.__bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        depth_img = self.__bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth_img = depth_img.astype(np.float32)

        try:
            output_depth, filtered_output_depth = self.__depthcomplete.depth_completion(
                color_img,
                depth_img,
                inertia_weight=float(self.__config.depth2depth.inertia_weight),
                smoothness_weight=float(self.__config.depth2depth.smoothness_weight),
                tangent_weight=float(self.__config.depth2depth.tangent_weight),
                mode_modify_input_depth=self.__config.modifyInputDepth.mode)
        except depth_completion_api.DepthCompletionError as e:
            rospy.logwarn('Depth Completion Failed:\n  {}'.format(e))
            return

        completed = self.__bridge.cv2_to_imgmsg(output_depth, encoding='passthrough')
        completed.header = depth_msg.header
        self.__pub.publish(completed)


if __name__ == '__main__':
    rospy.init_node('depth_completion')
    DepthCompletion()
    rospy.spin()
