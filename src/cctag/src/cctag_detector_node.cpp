#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <vision_msgs/msg/object_hypothesis.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>

#include <boost/ptr_container/ptr_list.hpp>

// CCTag
#include <cctag/ICCTag.hpp>  // provides cctag::ICCTag and cctag::cctagDetection()

class CCTagDetectorNode : public rclcpp::Node
{
public:
  CCTagDetectorNode()
  : Node("cctag_detector_node")
  {
    image_topic_ = this->declare_parameter<std::string>("image_topic", "/camera/color/image_raw");
    frame_id_override_ = this->declare_parameter<std::string>("frame_id_override", "");

    n_rings_ = static_cast<std::size_t>(this->declare_parameter<int>("n_rings", 3));
    pipe_id_ = this->declare_parameter<int>("pipe_id", 0);
    use_cuda_ = this->declare_parameter<bool>("use_cuda", false);

    detections_topic_ = this->declare_parameter<std::string>("detections_topic", "cctag/detections");

    det_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>(
      detections_topic_, rclcpp::QoS(10));

    // SensorDataQoS is typical for camera streams.
    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      image_topic_, rclcpp::SensorDataQoS(),
      std::bind(&CCTagDetectorNode::onImage, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Subscribing: %s", image_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Publishing:  %s", detections_topic_.c_str());
  }

private:
  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      // Accept common camera encodings. OAK-D often publishes rgb8 or bgr8.
      cv_ptr = cv_bridge::toCvShare(msg, msg->encoding);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN(get_logger(), "cv_bridge conversion failed: %s", e.what());
      return;
    }

    const cv::Mat & src = cv_ptr->image;
    if (src.empty()) {
      return;
    }

    cv::Mat gray;
    if (msg->encoding == "mono8" || msg->encoding == "8UC1") {
      gray = src;
    } else if (msg->encoding == "rgb8") {
      cv::cvtColor(src, gray, cv::COLOR_RGB2GRAY);
    } else {
      // Default to BGR->GRAY for typical OpenCV camera pipelines.
      cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    }

    // Run CCTag detection:
    // CCTag docs show using cctag::Parameters and cctag::cctagDetection(...) to fill a marker list. :contentReference[oaicite:1]{index=1}
    cctag::Parameters params(n_rings_);
    params.setUseCuda(use_cuda_);

    boost::ptr_list<cctag::ICCTag> markers;
    const std::size_t frame_number = frame_counter_++;
    cctag::cctagDetection(markers, pipe_id_, frame_number, gray, params);

    // Publish as vision_msgs/Detection2DArray. :contentReference[oaicite:2]{index=2}
    vision_msgs::msg::Detection2DArray out;
    out.header = msg->header;
    if (!frame_id_override_.empty()) {
      out.header.frame_id = frame_id_override_;
    }

    out.detections.clear();
    out.detections.reserve(std::distance(markers.begin(), markers.end()));

    for (const auto & marker : markers) {
      // ICCTag.hpp states only markers with status == 1 are valid. :contentReference[oaicite:3]{index=3}
      if (marker.getStatus() != 1) {
        continue;
      }

      vision_msgs::msg::Detection2D det;
      det.header = out.header;

      // Use the tag ID as a stable identifier string.
      const std::string tag_id = std::to_string(marker.id());
      det.id = tag_id;

      // Fill "results" with a single hypothesis.
      // In ROS 2 vision_msgs (ros2 branch), ObjectHypothesis has class_id (string) and score. :contentReference[oaicite:4]{index=4}
      vision_msgs::msg::ObjectHypothesisWithPose hyp;
      hyp.hypothesis.class_id = tag_id;
      hyp.hypothesis.score = 1.0;
      // Pose is unused here (we only publish 2D). Leave default.
      det.results.push_back(hyp);

      // BoundingBox2D: centre (Pose2D in pixel coords) and size_x/size_y. :contentReference[oaicite:5]{index=5}
      const auto & e = marker.rescaledOuterEllipse();  // example usage shows a(), b(), angle() :contentReference[oaicite:6]{index=6}
      det.bbox.center.position.x = static_cast<double>(marker.x());
      det.bbox.center.position.y = static_cast<double>(marker.y());
      det.bbox.center.theta = static_cast<double>(e.angle());  // radians
      det.bbox.size_x = 2.0 * static_cast<double>(e.a());      // ellipse diameter as box extent
      det.bbox.size_y = 2.0 * static_cast<double>(e.b());

      out.detections.push_back(std::move(det));
    }

    det_pub_->publish(out);
  }

  // Params
  std::string image_topic_;
  std::string detections_topic_;
  std::string frame_id_override_;
  std::size_t n_rings_{3};
  int pipe_id_{0};
  bool use_cuda_{false};

  // State
  std::atomic<std::size_t> frame_counter_{0};

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CCTagDetectorNode>());
  rclcpp::shutdown();
  return 0;
}