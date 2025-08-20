# MIT License
# This file is part of a project licensed under the MIT License.
# See the LICENSE file in the repository for details.

from time import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesis, BoundingBox2D

from ultralytics import YOLO
from cv_bridge import CvBridge

import numpy as np
import torch


class ImageDetectorNode(Node):
    """
    ROS 2 node that performs 2D object detection on incoming camera images using a YOLO model.

    It publishes:
    - The output image with bounding boxes.
    - Detections of humans
    """

    def __init__(self) -> None:
        super().__init__('image_detector_node')

        # Subscription to input images
        self.declare_parameter('input_topic', '/camera/image_raw')
        input_topic = self.get_parameter('input_topic').value
        self.create_subscription(Image, input_topic, self._main_callback, 1)

        # Publisher for detections
        self.declare_parameter('detections_topic', '/yolo/detections')
        detections_topic = self.get_parameter('detections_topic').value
        self._publisher = self.create_publisher(Detection2DArray, detections_topic, 1)

        # YOLO model parameters
        self.declare_parameter('YOLO_model', 'yolov5su.pt')
        self.declare_parameter('YOLO_threshold', 0.5)
        self.declare_parameter('device', 'cpu')

        YOLO_model = self.get_parameter('YOLO_model').value
        device = self.get_parameter('device').value
        self.threshold = self.get_parameter('YOLO_threshold').value
        self.classes = [0]  # Pedestrian (0)

        self.model = YOLO(YOLO_model).to(torch.device(device))
        self.model.fuse()

        self.bridge = CvBridge()

        self.get_logger().info("image_detector_node is up and running.")

    def _main_callback(self, image_msg: Image) -> None:
        """
        Callback function for the subscribed camera images.
        Performs detection and publishes processed data.
        """
        header = image_msg.header
        cv2_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')

        start = time()
        predictions = self.detect(cv2_image)
        elapsed = time() - start

        detections_msg = self.npbbox2detections(predictions, header)

        self._publisher.publish(detections_msg)

        self.get_logger().info(f"{len(predictions)} humans detected in {elapsed:.4f} seconds.")

    def detect(self, img: np.ndarray) -> np.ndarray:
        """
        Performs object detection on the input image.

        Args:
            img (np.ndarray): Input image in BGR format.

        Returns:
            - Numpy array of detections (bounding boxes, confidence, class_id).
        """
        predictions = self.model(
            img,
            conf=self.threshold,
            classes=self.classes,
            verbose=False
        )

        pred_bboxes = (predictions[0].boxes.data).detach().cpu().numpy()

        return pred_bboxes

    def npbbox2detections(self, detections: np.ndarray, header) -> Detection2DArray:
        """
        Converts numpy bounding boxes into ROS 2 Detection2DArray messages.

        Args:
            detections (np.ndarray): Array of detected bounding boxes.
            header: ROS message header to assign to outputs.

        Returns:
            detections_msg.
        """

        detections_msg = Detection2DArray()
        detections_msg.header = header

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
            hypo = ObjectHypothesis()
            hypo.hypothesis.class_id = str(class_id)
            hypo.hypothesis.score = float(conf)
            detection.results.append(hypo)

            # Classify detection
            detectionss_msg.detections.append(detection)

        return detections_msg


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
