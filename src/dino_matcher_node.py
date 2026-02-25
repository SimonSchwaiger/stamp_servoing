
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import v2 as T_v2

from PIL import Image

import contextlib

from typing import Dict, Any

enable_amp_autocast = True

_target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if _target_device == "cuda": _target_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else: _target_dtype = torch.float16

_maybe_autocast = (
        lambda: torch.amp.autocast(device_type=str(_target_device), dtype=_target_dtype)
        if enable_amp_autocast else contextlib.nullcontext() )

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

## TODO: Maybe we want the OTASv2 Featuriser here so that we can add negative prompts
class featuriser(nn.Module):
    def __init__(self, input_size = 518):
        super().__init__()

        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        self.backbone.to(_target_dtype).eval().to(_target_device)
        #self.backbone.compile(fullgraph=True, dynamic=True, options={"triton.cudagraphs": True})

        self.patch_feat_size = int(input_size/self.backbone.patch_size)

        self.compose_v2 = T_v2.Compose([
            T_v2.ToImage(),
            T_v2.ToDtype(_target_dtype, scale=True),
            T_v2.Resize((input_size, input_size), antialias=True),
            T_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def t_dino(self, input):
        # Takes single PIL image, tensor or list of images or list of tensors
        if isinstance(input, list):
            return torch.stack([self.compose_v2(img) for img in input]).to(_target_device)
        inp_tensor = self.compose_v2(input).to(_target_device)
        if len(inp_tensor.shape) < 4: inp_tensor = inp_tensor.unsqueeze(0)
        return inp_tensor

    @torch.inference_mode()
    def forward(self, x): 
        with _maybe_autocast():
            feat = self.backbone.forward_features(self.t_dino(x))["x_norm_patchtokens"]
            B, grid, C = feat.shape
            feat = feat.view(B, self.patch_feat_size, self.patch_feat_size, C)
            return feat.permute(0, 3, 1, 2) # B, C, W, H

    @torch.inference_mode()
    def forward_cls(self, x): 
        with _maybe_autocast():
            return self.backbone.forward_features(self.t_dino(x))["x_norm_clstoken"] # torch.Size([B, 384])

# See: https://github.com/RogerQi/maskclip_onnx/blob/main/clip_playground.ipynb
@torch.no_grad()
def clip_similarity(img_features: torch.Tensor, text_feats: torch.Tensor) -> Dict[str, Any]:
    return torch.einsum("chw,c->hw", F.normalize(img_features[0], dim=0), F.normalize(text_feats, dim=0))

@torch.no_grad()
def clip_similarity_flat(img_features_flat: torch.Tensor, text_feats: torch.Tensor) -> Dict[str, Any]:
    return torch.einsum("nc,c->n", F.normalize(img_features_flat, dim=1), F.normalize(text_feats, dim=0))


## Startup
# Init featuriser
# Template img cls token (this will be the main source of similarity)
# Patch token of target image 
# clip similarity (target image, cls token)
# Convert max of similarity to image space coordinates to derive the goal point


# model = featuriser()
# img_path = "/app/testdata/cluttered_grasp.png"
# img = Image.open(img_path).convert("RGB")
# feat = model(img)
featuriser = featuriser()
template_img_path = current_dir + "/imgs/template.png"
img = Image.open(template_img_path).convert("RGB")
template_cls_token = featuriser.forward_cls(img)[0]





import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as ROS_Image
from cv_bridge import CvBridge

class ImageProcNode(Node):
    def __init__(self):
        super().__init__('dino_matcher_node')
        self._bridge = CvBridge()

        self._sub = self.create_subscription(
            ROS_Image,
            '/oak/rgb/image_raw',
            self._image_cb,
            1,
        )
        self._pub = self.create_publisher(ROS_Image, 'image_modified', 1)

    def _image_cb(self, msg):
        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        # Modify cv_image here, then publish

        img = Image.fromarray(cv_image)
        patch_token = featuriser.forward(img)
        sim = clip_similarity(patch_token, template_cls_token)
        # Upscale patch-level similarity to image size (H, W)
        sim_resized = F.interpolate(
            sim.unsqueeze(0).unsqueeze(0),
            size=img.size[::-1],  # (height, width)
            mode="nearest",
        ).squeeze().cpu().numpy()

        # Normalize similarity to [0, 1] for overlay
        s_min, s_max = sim_resized.min(), sim_resized.max()
        sim_norm = (sim_resized - s_min) / (s_max - s_min + 1e-8)
        sim_threshold = 0.6
        sim_norm[sim_norm < sim_threshold] = 0

        # Highlight high similarity in the red channel
        # With passthrough: rgb8 -> red=0, bgr8 -> red=2
        red_idx = 0 if msg.encoding == "rgb8" else 2
        red_strength = 1.0  # blend strength for the similarity overlay
        red_channel = cv_image[:, :, red_idx].astype(np.float32)
        red_channel = np.clip(
            red_channel + red_strength * sim_norm * 255, 0, 255
        ).astype(np.uint8)
        cv_image[:, :, red_idx] = red_channel

        out_msg = self._bridge.cv2_to_imgmsg(cv_image, encoding=msg.encoding)
        out_msg.header = msg.header
        self._pub.publish(out_msg)


if __name__ == '__main__':
    rclpy.init()
    node = ImageProcNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
