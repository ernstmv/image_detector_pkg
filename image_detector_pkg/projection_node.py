import rclpy
from rclpy.node import Node

from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose, BoundingBox2D
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import TimeSynchronizer, Subscriber

import cv2
import numpy as np


class ProjectionNode(Node):
    
    def __init__(self):
        super().__init__('projection_node')

        # SUBSCRIPTIONS
        self.declare_parameter('detections_topic', '/yolo/detections')
        detections_topic = self.get_parameter('detections_topic').value

        self.declare_parameter('input_image_topic', '/input_image')
        input_image_topic = self.get_parameter('input_image_topic').value

        ts = TimeSynchronizer(

            [
                Subscriber(self, Detection2DArray, detections_topic),
                Subscriber(self, Image, input_image_topic)
            ],

            queue_size=10
        )

        ts.registerCallback(self._main_pipeline)

        # PUBLISHER
        self.declare_parameter('output_image_topic', '/yolo/image')
        output_image_topic = self.get_parameter('output_image_topic').value
        self.publisher = self.create_publisher(Image, output_image_topic, 1)

        self.bridge = CvBridge()

        self.get_logger().info("Projection node working.")

    def _main_pipeline(self, detections_msg: Detection2DArray, image_msg: Image) -> None:
        cv_image = self._imgmsg2np(image_msg)

        cv_image_detections = self._draw_detections(cv_image, detections_msg)

        image_msg = self._np2imgmsg(cv_image_detections)
        self.publisher.publish(image_msg)

    def _draw_detections(self, image: np.ndarray, detections_msg: Detection2DArray) -> np.ndarray:
        """
        Draw bounding boxes and class labels on the image, ensuring coordinates stay within image bounds.

        Args:
            image (np.ndarray): Input image (H x W x C, BGR format).
            detections_msg (Detection2DArray): ROS 2 message with detections.

        Returns:
            np.ndarray: Image with drawn detections.
        """
        output_image = image.copy()
        height, width = output_image.shape[:2]

        for detection in detections_msg.detections:
            bbox = detection.bbox
            # Convert center + size to top-left and bottom-right
            x_center = int(bbox.center.position.x)
            y_center = int(bbox.center.position.y)
            w = int(bbox.size_x)
            h = int(bbox.size_y)

            x1 = max(0, x_center - w // 2)
            y1 = max(0, y_center - h // 2)
            x2 = min(width - 1, x_center + w // 2)
            y2 = min(height - 1, y_center + h // 2)

            # Draw rectangle
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

            # Draw class label + score (first hypothesis)
            if detection.results:
                hypothesis = detection.results[0].hypothesis
                class_id = hypothesis.class_id
                score = hypothesis.score
                label = f"{class_id}: {score:.2f}"

                # Ensure text is inside image
                text_y = max(0, y1 - 5)
                cv2.putText(
                    output_image,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )

        return output_image

    def _imgmsg2np(self, img_msg: Image) -> np.ndarray:
        """
        Convert a sensor_msgs/Image to a numpy array.

        Args:
            img_msg (Image): Image message.

        Returns:
            np.ndarray: OpenCV image.
        """
        return self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')

    def _np2imgmsg(self, arr: np.ndarray) -> Image:
        """
        Convert a numpy array to a sensor_msgs/Image message.

        Args:
            arr (np.ndarray): OpenCV image.

        Returns:
            Image: Image message.
        """
        return self.bridge.cv2_to_imgmsg(arr, encoding='bgr8')

def main(args=None) -> None:
    """
    Initializes and spins the ImageDetectorNode.
    """
    rclpy.init(args=args)
    node = ProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
