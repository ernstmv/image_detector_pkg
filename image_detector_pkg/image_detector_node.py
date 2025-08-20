# MIT License
# This file is part of a project licensed under the MIT License.
# See the LICENSE file in the repository for details.

from time import time
from typing import Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from ultralytics import YOLO
from cv_bridge import CvBridge

import numpy as np
import torch


class ImageDetectorNode(Node):
    """
    ROS 2 node that performs 2D object detection on incoming camera images using a YOLO model.

    It publishes:
    - The output image with bounding boxes.
    - Detections of humans and cars separately.
    """

    def __init__(self) -> None:
        super().__init__('image_detector_node')

        # Subscription to input images
        self.declare_parameter('input_topic', '/camera/image_raw')
        input_topic = self.get_parameter('input_topic').value
        self.create_subscription(Image, input_topic, self._image_callback, 1)

        # Publishers for processed images and detections
        self.declare_parameter('output_image_topic', '/yolo/image')
        self.declare_parameter('human_detections_topic', '/yolo/human_detections_2d')
        self.declare_parameter('car_detections_topic', '/yolo/car_detections_2d')

        output_image_topic = self.get_parameter('output_image_topic').value
        human_detections_topic = self.get_parameter('human_detections_topic').value
        car_detections_topic = self.get_parameter('car_detections_topic').value

        self.image_publisher = self.create_publisher(Image, output_image_topic, 1)
        self.humans_publisher = self.create_publisher(Detection2DArray, human_detections_topic, 1)
        self.cars_publisher = self.create_publisher(Detection2DArray, car_detections_topic, 1)

        # YOLO model parameters
        self.declare_parameter('YOLO_model', 'yolov5su.pt')
        self.declare_parameter('YOLO_threshold', 0.5)
        self.declare_parameter('device', 'cpu')

        YOLO_model = self.get_parameter('YOLO_model').value
        device = self.get_parameter('device').value
        self.threshold = self.get_parameter('YOLO_threshold').value
        self.classes = [0, 2]  # Pedestrian (0) and Car (2)

        self.model = YOLO(YOLO_model).to(torch.device(device))
        self.model.fuse()

        self.bridge = CvBridge()
        self.get_logger().info("2D detection node is up and running.")

    def _image_callback(self, image_msg: Image) -> None:
        """
        Callback function for the subscribed camera images.
        Performs detection and publishes processed data.
        """
        header = image_msg.header
        cv2_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        start = time()
        output_image, predictions = self.detect(cv2_image)
        elapsed = time() - start

        output_image_msg = self.bridge.cv2_to_imgmsg(output_image, encoding='bgr8')
        car_pred_msg, human_pred_msg = self.npbbox2detections(predictions, header)

        self.image_publisher.publish(output_image_msg)
        self.humans_publisher.publish(human_pred_msg)
        self.cars_publisher.publish(car_pred_msg)

        self.get_logger().info(f"{len(predictions)} objects detected in {elapsed:.4f} seconds.")

    def detect(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs object detection on the input image.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            Tuple containing:
                - Processed image with bounding boxes drawn.
                - Numpy array of detections (bounding boxes, confidence, class_id).
        """
        predictions = self.model(
            img,
            conf=self.threshold,
            classes=self.classes,
            verbose=False
        )

        img_with_boxes = predictions[0].plot()
        pred_bboxes = (predictions[0].boxes.data).detach().cpu().numpy()

        return img_with_boxes, pred_bboxes

    def npbbox2detections(self, detections: np.ndarray, header) -> Tuple[Detection2DArray, Detection2DArray]:
        """
        Converts numpy bounding boxes into ROS 2 Detection2DArray messages.

        Args:
            detections (np.ndarray): Array of detected bounding boxes.
            header: ROS message header to assign to outputs.

        Returns:
            Tuple of (cars_detections_msg, humans_detections_msg).
        """
        cars_msg = Detection2DArray()
        cars_msg.header = header

        humans_msg = Detection2DArray()
        humans_msg.header = header

        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            detection = Detection2D()

            # Define bounding box
            bbox = BoundingBox2D()
            bbox.center.position.x = float((x1 + x2) / 2.0)
            bbox.center.position.y = float((y1 + y2) / 2.0)
            bbox.size_x = float(x2 - x1)
            bbox.size_y = float(y2 - y1)
            detection.bbox = bbox

            # Define object hypothesis
            hypo = ObjectHypothesisWithPose()
            hypo.hypothesis.class_id = str(class_id)
            hypo.hypothesis.score = float(conf)
            detection.results.append(hypo)

            # Classify detection
            if hypo.hypothesis.class_id == '0.0':
                humans_msg.detections.append(detection)
            elif hypo.hypothesis.class_id == '2.0':
                cars_msg.detections.append(detection)

        return cars_msg, humans_msg


def main(args=None) -> None:
    """
    Initializes and spins the ImageDetectorNode.
    """
    rclpy.init(args=args)
    node = ImageDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
