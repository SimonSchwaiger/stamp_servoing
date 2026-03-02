from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.layers import use_fused_attn
#from torchvision import transforms

from torchvision.transforms import v2 as T_v2

import numpy as np
import math

import contextlib
from typing import Union, List, Dict, Any

from PIL import Image

enable_amp_autocast = True

_target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if _target_device == "cuda": _target_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else: _target_dtype = torch.float16

_maybe_autocast = (
        lambda: torch.amp.autocast(device_type=str(_target_device), dtype=_target_dtype)
        if enable_amp_autocast else contextlib.nullcontext() )

# This is bad but required as open_clip_torch just tries to access this fn regardless if available. Seems to be old pytorch behaviour
# pip install timm open_clip_torch einops
on_jetson = False
if on_jetson: torch.distributed.is_initialized = lambda: False 

## GaussKernelAttn from NARADIO. See: https://github.com/RayFronts/RayFronts/blob/main/rayfronts/image_encoders/naradio.py
class GaussKernelAttn(nn.Module):
  """Encompases the NACLIP attention mechanism."""

  def __init__(
    self,
    orig_attn,
    input_resolution: tuple,
    gauss_std: float,
    device,
    chosen_cls_id: int,
    dim: int,
    qk_norm: bool = False,
    num_prefix_tokens: int = 8,
  ) -> None:
    super().__init__()
    num_heads = orig_attn.num_heads
    assert dim % num_heads == 0, "dim should be divisible by num_heads"
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5
    self.fused_attn = use_fused_attn()
    self.input_resolution = input_resolution

    h, w = input_resolution
    n_patches = (w // 16, h //16)
    window_size = [side * 2 - 1 for side in n_patches]
    window = GaussKernelAttn.gaussian_window(*window_size, std=gauss_std,
                                             device=device)
    # Register as buffer so compiler can trace it
    self.register_buffer('attn_addition', GaussKernelAttn.get_attention_addition(
      *n_patches, window, num_prefix_tokens
    ).unsqueeze(0))

    self.chosen_cls_id = chosen_cls_id
    self.gauss_std = gauss_std

    self.qkv = orig_attn.qkv
    self.q_norm = orig_attn.q_norm if qk_norm else nn.Identity()
    self.k_norm = orig_attn.k_norm if qk_norm else nn.Identity()
    self.attn_drop = orig_attn.attn_drop
    self.proj = orig_attn.proj
    self.proj_drop = orig_attn.proj_drop
    self.device = device
    self.num_prefix_tokens = num_prefix_tokens

  def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    B, N, C = x.shape
    x_out = self._custom_attn_compiled(x.permute(1, 0, 2))
    x_out = x_out.permute(1, 0, 2)
    return x_out

  @staticmethod
  def gaussian_window(dim1, dim2, std=5., device="cuda"):
    constant = 1 / (std * math.sqrt(2))
    start = -(dim1 - 1) / 2.0
    k1 = torch.linspace(start=start * constant,
                        end=(start + (dim1 - 1)) * constant,
                        steps=dim1,
                        dtype=torch.float, device=device)
    start = -(dim2 - 1) / 2.0
    k2 = torch.linspace(start=start * constant,
                        end=(start + (dim2 - 1)) * constant,
                        steps=dim2,
                        dtype=torch.float, device=device)
    dist_square_to_mu = (torch.stack(torch.meshgrid(
      k1, k2, indexing="ij")) ** 2).sum(0)

    return torch.exp(-dist_square_to_mu)

  @staticmethod
  def get_attention_addition(dim1, dim2, window, num_prefix_tokens=8):
    d = window.device
    m = torch.einsum("ij,kl->ijkl",
                     torch.eye(dim1, device=d),
                     torch.eye(dim2, device=d))
    m = m.permute((0, 3, 1, 2)).contiguous()
    out = F.conv2d(m.view(-1, dim1, dim2).unsqueeze(1),
                   window.unsqueeze(0).unsqueeze(1),
                   padding='same').squeeze(1)

    out = out.view(dim1 * dim2, dim1 * dim2)
    if num_prefix_tokens > 0:
      v_adjusted = torch.vstack(
        [torch.zeros((num_prefix_tokens, dim1 * dim2), device=d), out])
      out = torch.hstack([torch.zeros(
        (dim1 * dim2 + num_prefix_tokens, num_prefix_tokens), device=d),
        v_adjusted])

    return out

  @torch.compile(mode="max-autotune", fullgraph=True)
  def _custom_attn_compiled(self, x: torch.Tensor) -> torch.Tensor:
    """Compiled attention - hot path for inference."""
    num_heads = self.num_heads
    num_tokens, bsz, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5

    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q, k = self.q_norm(q), self.k_norm(k)

    q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    # kk.T vs kq.T has the most impact
    attn_weights = torch.bmm(k, k.transpose(1, 2)) * scale

    # Gaussian attention addition
    attn_weights = attn_weights + self.attn_addition
    attn_weights = F.softmax(attn_weights, dim=-1)

    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(
      -1, bsz, embed_dim)
    attn_output = self.proj(attn_output)
    attn_output = self.proj_drop(attn_output)

    return attn_output

  def update_input_resolution(self, input_resolution):
    """Update attn_addition buffer for new resolution. Note: triggers recompilation."""
    h, w = input_resolution
    n_patches = (w // 16, h //16)
    window_size = [side * 2 - 1 for side in n_patches]
    window = GaussKernelAttn.gaussian_window(*window_size, std=self.gauss_std,
                                             device=self.device)
    # Update the registered buffer
    self.register_buffer('attn_addition', GaussKernelAttn.get_attention_addition(
      *n_patches, window, self.num_prefix_tokens
    ).unsqueeze(0))

class RadioFeaturizer(nn.Module):
    def __init__(self, target_hw: int = 512, compile=False):
        super().__init__()

        RADIO_VERSION = 'c-radio_v3-b'
        TARGET_HW = (target_hw, target_hw) 
        self.adaptor_name = "clip"
        enable_gaussian_attn = True
        self.alpha = 0. # MaskCLIP shift gain [0.5-1.0]
        self.model = torch.hub.load(
            'NVlabs/RADIO',
            'radio_model',
            version=RADIO_VERSION,
            progress=True,
            skip_validation=True,
            adaptor_names=self.adaptor_name
        )
        self.model.to(torch.bfloat16).eval().to(_target_device) # Run at half precision to save VRAM
        #self.model.make_preprocessor_external()
        
        ## Setup customised lang adapter
        self.lang_adaptor = self.model.adaptors[self.adaptor_name]
        self.model.adaptors = None
        cutoff = -3 # When to apply MaskCLIP distribution shift
        self.clip_summary_proj = self.lang_adaptor.head_mlp.final[cutoff:] # Extract last projection
        self.lang_adaptor.head_mlp.final = self.lang_adaptor.head_mlp.final[:cutoff] # Remove last projection for maskCLIP
        self.clip_feat_proj = self.lang_adaptor.feat_mlp.final[cutoff:] # Extract last projection
        self.lang_adaptor.feat_mlp.final = self.lang_adaptor.feat_mlp.final[:cutoff] # Remove last projection for maskCLIP
        
        ## NARADIO Gaussian Kernel Attention
        if enable_gaussian_attn == True:
            last_block = self.model.model.blocks[-1]
            last_block.attn = GaussKernelAttn(
                last_block.attn,
                TARGET_HW,
                0.7,
                dim=self.model.model.embed_dim,
                chosen_cls_id=self.lang_adaptor.head_idx,
                device=_target_device,
                num_prefix_tokens=self.model.num_summary_tokens)
        
        # Preprocessing: ToTensor gives [0,1] floats. RADIO handles normalization itself
        self.compose_v2 = T_v2.Compose([
            T_v2.Resize(TARGET_HW, antialias=True),
            T_v2.ToImage(),
            T_v2.ToDtype(_target_dtype, scale=True),
        ])

        # Cache these values in __init__ to avoid module inspection in forward (graph break)
        self.C_in = self._first_linear_in_features(self.lang_adaptor.feat_mlp)  # e.g. 1024
        self.head_idx = int(getattr(self.lang_adaptor, "head_idx", 0))
        
        if compile:
            self.model.compile(fullgraph=True, dynamic=True, options={"triton.cudagraphs": True})
            self.lang_adaptor.compile(fullgraph=True, dynamic=True, options={"triton.cudagraphs": True})

    def t_radio(self, input: Union[list, Image.Image, torch.Tensor]):
        # Takes single PIL image, tensor or list of images or list of tensors
        if isinstance(input, Image.Image): input = [input] # Auto expand to batch size 1

        if isinstance(input, list):
            return torch.stack([self.compose_v2(img) for img in input]).to(_target_device) # Stack as tensors on correct device

        return self.compose_v2(input).to(_target_device)

    @staticmethod
    def _first_linear_in_features(m: nn.Module) -> int:
        for sub in m.modules():
            if isinstance(sub, nn.Linear):
                return sub.in_features
        raise RuntimeError("feat_mlp has no nn.Linear")

    @torch.inference_mode()
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # Img shape B, 3, H, W
        with _maybe_autocast():
            radio_output = self.model(img, feature_fmt='NCHW')
            bb_summary  = radio_output.summary           # [B, C_tot]
            bb_features = radio_output.features          # [B, C_tot, H, W]
            
            B, C_tot = bb_summary.shape
            assert C_tot % self.C_in == 0, f"C_tot={C_tot} not divisible by C_in={self.C_in}"
            n_heads = C_tot // self.C_in
            assert 0 <= self.head_idx < n_heads, f"head_idx {self.head_idx} out of range (n_heads={n_heads})"

            # Extract lang head and patch token
            start, end = self.head_idx * self.C_in, (self.head_idx + 1) * self.C_in
            cls_pre   = bb_summary[:, start:end]                 # [B, C_in]
            feats_pre = bb_features[:, start:end, ...]           # [B, C_in, H, W]
            B, C_in, H, W = feats_pre.shape
            patches_tok   = feats_pre.permute(0, 2, 3, 1).reshape(B, H * W, C_in)  # [B, N, C_in]

            # Project both forward right before the linear projection
            cls_pre = self.lang_adaptor.head_mlp(cls_pre)
            patches_tok = self.lang_adaptor.head_mlp(patches_tok)

            ## With only the last linear layer remaining, perform maskclip distribution shift
            mean_u    = patches_tok.mean(dim=1, keepdim=True)
            patches_tok = patches_tok + self.alpha * (cls_pre.unsqueeze(1) - mean_u)

            ## Project to lang and normalise
            cls_lang = self.clip_summary_proj(cls_pre)
            patches_lang = self.clip_summary_proj(patches_tok)

            patches_lang = F.normalize(patches_lang, dim=-1)
            cls_lang     = F.normalize(cls_lang, dim=-1)

            # Pack dense map as NCHW for downstream use
            clip_features = patches_lang.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, C_out, H, W]
            clip_summary  = cls_lang                                                          # [B, C_out]

            output = {
                self.adaptor_name: (clip_summary, clip_features),
                "backbone": (bb_summary, bb_features) }

            return output

    @torch.inference_mode()
    def clip_encode_text(self, text: str) -> Dict[str, Any]:
        with _maybe_autocast():
            tok = self.lang_adaptor.tokenizer(text).to(_target_device)
            t = self.lang_adaptor.encode_text(tok)  # shape (B,T,D_txt) or (B,D_txt)

            if t.ndim == 3:
                t = t[:, -1, :]  # EOS / last token
            else:
                t = t  # (B,D_txt)

            t = F.normalize(t.squeeze(0), dim=-1)
            return t #TODO: Research what intermediate token do in open_clip

    @torch.inference_mode()
    def clip_pipe_global_token(self, img: Image.Image) -> Dict[str, Any]:
        output = self.forward(self.t_radio(img)) # This auto-batches
        return output[self.adaptor_name][0][0] # Returns language summary token of the "first image" (there is only one)
    
    @torch.no_grad()
    def clip_global_feat_switch(self, prompt: Union[str, Image.Image, List]) -> Dict[str, Any]:
        """Automatically switches between text and global image features for CLIP queries."""
        if isinstance(prompt, (Image.Image, List)): return self.clip_pipe_global_token(prompt)
        else: return self.clip_encode_text(prompt)

class TorchKMeans(nn.Module):
    def __init__(self, n_clusters: int, max_iter: int = 10):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        # Don't use register_buffer for None - causes issues with compile guards
        self.centroids_: torch.Tensor | None = None
        self._is_fitted = False

    def fit(self, X: torch.Tensor):
        """Fit KMeans - vectorized, called once, not compiled."""
        N, D = X.shape
        K = self.n_clusters
        
        # Use generator instead of global seed (avoids graph break)
        g = torch.Generator(device=X.device).manual_seed(42)
        centroids = X[torch.randperm(N, generator=g, device=X.device)[:K]].clone()

        for _ in range(self.max_iter):
            # Assign labels
            dists = torch.cdist(X, centroids, p=2)  # [N, K]
            labels = dists.argmin(dim=1)            # [N]
            
            # Vectorized centroid update (no Python loop over clusters)
            one_hot = F.one_hot(labels, K).to(X.dtype)  # [N, K]
            counts = one_hot.sum(dim=0, keepdim=True).T  # [K, 1]
            counts = counts.clamp(min=1)
            centroids = (one_hot.T @ X) / counts         # [K, D]

        self.centroids_ = centroids
        self._is_fitted = True

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict cluster labels. Compiled after few_shot_pretrain if used."""
        dists = torch.cdist(X, self.centroids_, p=2)
        return dists.argmin(dim=1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Auto-fits if not fitted, then predicts."""
        if not self._is_fitted:
            self.fit(X)
        return self.predict(X)

class TorchPCA(nn.Module):
    def __init__(self, n_components: int, ignore_first_components: int = 0):
        super().__init__()
        self.n_components = n_components
        self.ignore_first_components = ignore_first_components
        # Don't use register_buffer for None - causes issues with compile guards
        self.mean_: torch.Tensor | None = None
        self.components_: torch.Tensor | None = None
        self._is_fitted = False
        
    def fit(self, X: torch.Tensor):
        """Fit PCA - called once, not compiled."""
        orig_dtype = X.dtype
        with torch.no_grad():
            self.mean_ = X.mean(dim=0)
            X_centered = X - self.mean_
            # SVD doesn't support half precision on CUDA - compute in float32
            _, _, Vh = torch.linalg.svd(X_centered.float(), full_matrices=False)
            start = self.ignore_first_components
            end = start + self.n_components
            # Cast back to original dtype for inference
            self.components_ = Vh[start:end].to(orig_dtype)
            self.mean_ = self.mean_.to(orig_dtype)
            self._is_fitted = True

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """Transform data. Compiled after few_shot_pretrain if used."""
        return (X - self.mean_) @ self.components_.T

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Auto-fits if not fitted, then transforms."""
        if not self._is_fitted:
            self.fit(X)
        return self.transform(X)

class ClusterModel(nn.Module):
    def __init__(self, n_clusters: int, n_components: int, dino_scale_factor: int):
        super().__init__()
        self.dino_scale_factor = dino_scale_factor

        self._dim_reduction_factory = lambda: TorchPCA(n_components=n_components)
        self._cluster_model_factory = lambda:  TorchKMeans(n_clusters=n_clusters)
        self._trained_dim_reduction_model = None # Placeholder for few-shot pretraining
        self._trained_cluster_model = None

    def _ensure_tensor(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        if isinstance(X, np.ndarray): X = torch.from_numpy(X)
        return X.to(_target_device)

    def _ensure_numpy(self, X: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        if isinstance(X, torch.Tensor): X = X.detach().cpu().numpy()
        return X

    def _dim_reduction(self, feature_list: Union[torch.Tensor, np.ndarray], store_output: bool = False) -> torch.Tensor:
        if self._trained_dim_reduction_model is None: 
            dim_reduction = self._dim_reduction_factory()
            reduced_features = dim_reduction.forward(feature_list)
        else: 
            if store_output: assert False, "Tried to store over already trained dim reduction model. Few-shot pretraining can only be done once."
            reduced_features = self._trained_dim_reduction_model.forward(feature_list)

        reduced_features = F.normalize(self._ensure_tensor(reduced_features), dim=-1)
        if store_output: self._trained_dim_reduction_model = dim_reduction
        return reduced_features
    
    def _cluster_model(self, reduced_features: Union[torch.Tensor, np.ndarray], store_output: bool = False) -> torch.Tensor:
        if self._trained_cluster_model is None:
            cluster_model = self._cluster_model_factory()
            cluster_labels = cluster_model.forward(reduced_features)  # Shape: (num_patches,), can be torch tensor on gpu or numpy array
        else:
            if store_output: assert False, "Tried to store over already trained cluster model. Few-shot pretraining can only be done once."
            cluster_labels = self._trained_cluster_model.forward(reduced_features)
            
        cluster_labels = self._ensure_tensor(cluster_labels)
        unique_labels = torch.unique(cluster_labels, sorted=True)
        if -1 in unique_labels: cluster_labels[cluster_labels == -1] = len(unique_labels)  # Assign a new cluster label for noise
        
        if store_output: self._trained_cluster_model = cluster_model
        return cluster_labels

    def few_shot_pretrain(self, ssl_embeddings: torch.Tensor, compile: bool = True) -> None:
        """Fit PCA and KMeans on embeddings, then optionally compile for fast inference."""
        _ = self.forward(ssl_embeddings, store_output=True)
        
        if compile:
            # Compile the stored instances' hot methods after fitting
            # This amortizes compilation cost over many subsequent inference calls
            self._trained_dim_reduction_model.transform = torch.compile(
                self._trained_dim_reduction_model.transform, mode="default" )
            self._trained_cluster_model.predict = torch.compile(
                self._trained_cluster_model.predict, mode="default" )

    @torch.no_grad()
    def forward(self, ssl_embeddings: torch.Tensor, store_output: bool = False) -> torch.Tensor:
        # In: Embeddings in shape B, D, H, W (Native to radio)
        B, D, H, W = ssl_embeddings.shape
        # Bilinearly scale based on dino_scale_factor
        H_ = H * self.dino_scale_factor
        W_ = W * self.dino_scale_factor
        ssl_embeddings = F.interpolate(ssl_embeddings, size=(H_, W_), mode="bilinear", align_corners=False)
        # Flatten to list of feats (B*H*W, D)
        ssl_embeddings = ssl_embeddings.permute(0, 2, 3, 1).reshape(B*H_*W_, D)
        # Dim reduction
        ssl_reduceds = self._dim_reduction(ssl_embeddings, store_output=store_output)
        # Cluster model
        cluster_labels = self._cluster_model(ssl_reduceds, store_output=store_output)
        # Reshape back to image
        return cluster_labels.reshape(B, H_, W_).unsqueeze(1) # B, 1, H_, W_ # Returns cluster_map

@torch.no_grad()
@torch.compile(mode="max-autotune")
def masked_average_pooling_loop(cluster_map: torch.Tensor, language_map: torch.Tensor) -> torch.Tensor:
    # img shape B, 3, H, W,
    # langauge_map shape B, D, scale*H/patch_size, scale*W/patch_size
    # cluster_map shape B, 1, H, W
    B_c, _, H_c, W_c = cluster_map.shape
    B_l, D_l, H_l, W_l = language_map.shape
    assert B_c == B_l, "Batch size must match between cluster map and language map"

    # For quick lookup
    dw = W_c/W_l; dh = H_c/H_l # ssl_embeddings scale factor * patch_size. Both (h&w) are kept here if non-quadratic patches
    #cw2lw = lambda w: int(w/dw); ch2lh = lambda h: int(h/dh)

    # Placeholder for result (more compact than full feat map)
    cluster_labels = torch.unique(cluster_map) # C
    feat_sums = torch.zeros((len(cluster_labels)+1, D_l), device=language_map.device, dtype=language_map.dtype) # (C+1, D_l)
    counts = torch.zeros((len(cluster_labels)+1, 1), device=language_map.device) #TODO: dtpe int here? # C+1

    # Indices of each cluster entry
    for idx, label in enumerate(cluster_labels):
    #label = 0; idx = 0
        indices = torch.nonzero(cluster_map == label)
        transformed_w = (indices[:,3]/dw).to(int)
        transformed_h = (indices[:,2]/dw).to(int)

        indices_t = torch.hstack((
            indices[:,0].unsqueeze(0).T, transformed_h.unsqueeze(0).T, transformed_w.unsqueeze(0).T))

        ## Deduplicate to not blow up vram
        unique_indices, counts_per_patch = torch.unique(indices_t, dim=0, return_counts=True)
        # Sum up deduplicated entries weighted by occurence
        b, h, w = unique_indices[:, 0], unique_indices[:, 1], unique_indices[:, 2]
        feats = language_map[b, :, h, w]   # (U, D)
        # Add to aggregated clusters
        feat_sums[idx] = (feats * counts_per_patch.unsqueeze(1).to(feats.dtype)).sum(dim=0)
        counts[idx] = indices_t.shape[0]

    # Store stuff needed to reconstruct feat map on the fly when we need it (storing it straight away takes all the vram)
    cluster_flat = cluster_map.permute(0, 2, 3, 1).reshape(B_c*H_c*W_c, 1)

    counts = counts.clamp(min=1); avg_feats = feat_sums / counts # No division by zero
    avg_feats = avg_feats / avg_feats.norm(dim=1, keepdim=True)
    output = avg_feats[cluster_flat]  # (N, C)
    return output.view(B_l, H_c, W_c, D_l).permute(0, 3, 1, 2)

#TODO: Add torch compile compatibility
@torch.no_grad()
def masked_average_pooling(cluster_map: torch.Tensor, language_map: torch.Tensor) -> torch.Tensor:
    """
    Vectorized masked average pooling: aggregate language features per cluster without a Python loop.
    cluster_map: (B, 1, H_c, W_c), language_map: (B, D, H_l, W_l).
    Returns: (B, D, H_c, W_c) with per-pixel cluster-averaged (and L2-normalized) language features.
    """
    B_c, _, H_c, W_c = cluster_map.shape
    B_l, D_l, H_l, W_l = language_map.shape
    assert B_c == B_l, "Batch size must match between cluster map and language map"

    dh = H_c / H_l
    dw = W_c / W_l
    num_lang_patches = B_l * H_l * W_l

    cluster_labels = torch.unique(cluster_map)
    num_clusters = len(cluster_labels)

    # Label -> index in [0, num_clusters-1] for indexing feat_sums/avg_feats
    L = torch.full(
        (cluster_labels.max().item() + 1,), -1, device=cluster_map.device, dtype=torch.long
    )
    L[cluster_labels] = torch.arange(num_clusters, device=cluster_map.device)

    language_map_flat = language_map.permute(0, 2, 3, 1).reshape(-1, D_l)

    # Per-pixel language linear index (vectorized over all pixels)
    b_idx = torch.arange(B_c, device=cluster_map.device).view(B_c, 1, 1).expand(B_c, H_c, W_c)
    h_l = (torch.arange(H_c, device=cluster_map.device).view(1, -1, 1) / dh).long().clamp(0, H_l - 1)
    w_l = (torch.arange(W_c, device=cluster_map.device).view(1, 1, -1) / dw).long().clamp(0, W_l - 1)
    lang_linear_flat = (b_idx * (H_l * W_l) + h_l * W_l + w_l).reshape(-1)

    cluster_flat = cluster_map.view(-1).squeeze(-1) if cluster_map.dim() == 4 else cluster_map.view(-1)
    cluster_idx = L[cluster_flat]

    combined_key = cluster_idx * num_lang_patches + lang_linear_flat
    unique_combined, counts_per = torch.unique(combined_key, return_counts=True)

    cluster_idx_u = unique_combined // num_lang_patches
    lang_linear_u = unique_combined % num_lang_patches

    feats = language_map_flat[lang_linear_u]
    counts_per = counts_per.to(feats.dtype)

    feat_sums = torch.zeros(
        (num_clusters, D_l), device=language_map.device, dtype=language_map.dtype
    )
    count_sums = torch.zeros((num_clusters, 1), device=language_map.device, dtype=feats.dtype)

    feat_sums.scatter_add_(
        0, cluster_idx_u.unsqueeze(1).expand(-1, D_l), feats * counts_per.unsqueeze(1)
    )
    count_sums.scatter_add_(0, cluster_idx_u.unsqueeze(1), counts_per.unsqueeze(1))

    count_sums = count_sums.clamp(min=1)
    avg_feats = feat_sums / count_sums
    avg_feats = avg_feats / avg_feats.norm(dim=1, keepdim=True)

    output_flat = avg_feats[cluster_idx]
    return output_flat.view(B_c, H_c, W_c, D_l).permute(0, 3, 1, 2)

# See: https://github.com/RogerQi/maskclip_onnx/blob/main/clip_playground.ipynb
@torch.no_grad()
def clip_similarity(img_features: torch.Tensor, text_feats: torch.Tensor) -> Dict[str, Any]:
    return torch.einsum("chw,c->hw", F.normalize(img_features[0].to(torch.float32), dim=0), F.normalize(text_feats, dim=0))

@torch.no_grad()
def clip_similarity_flat(img_features_flat: torch.Tensor, text_feats: torch.Tensor) -> Dict[str, Any]:
    return torch.einsum("nc,c->n", F.normalize(img_features_flat, dim=1), F.normalize(text_feats, dim=0))

@torch.no_grad()
def similarity(shared_feature_map: torch.Tensor, pos_prompts: List[Union[str, Image.Image]], neg_prompts: List[str] = [""]) -> torch.Tensor:
    img_feats = shared_feature_map.permute(2, 0, 1).unsqueeze(0)
    positive_feats = [ featuriser.clip_global_feat_switch(prompt) for prompt in pos_prompts ]
    pos_sims = [ clip_similarity(img_feats, text_feats) for text_feats in positive_feats ]

    if neg_prompts != [""]:
        negative_feats = [ featuriser.clip_global_feat_switch(prompt) for prompt in neg_prompts ]
        neg_sims = [ clip_similarity(img_feats, text_feats) for text_feats in negative_feats ]
    else: neg_sims = torch.zeros_like(pos_sims[0]) # This uses the same device as pos_sims

    lr_sims = sum(pos_sims)/len(pos_sims) - sum(neg_sims)/(len(neg_sims) + 1e-8) # Weight sum here to deal with len(pos) != len(neg)

    sim_min, sim_max = lr_sims.min(), lr_sims.max()
    sim_range = torch.clamp(sim_max - sim_min, min=0.1) # Prevent normalisation from blowing up noise due to low similarity
    lr_sims_norm = (lr_sims - sim_min) / (sim_range + 1e-8)
    
    return lr_sims_norm

#############
## ENCODER SETUP

import os
current_dir = os.path.dirname(os.path.abspath(__file__))

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


img_size = 512; pretrain=False
featuriser = RadioFeaturizer(target_hw=img_size).to(_target_dtype).eval()
cluster_model = ClusterModel(n_clusters=14, n_components=8, dino_scale_factor=2).to(_target_dtype).eval()

template_img_path = current_dir + "/imgs/raw_card_backup.png"
img = Image.open(template_img_path).convert("RGB").resize((img_size, img_size))

# Warmup pass (triggers compilation)
with torch.no_grad():
    warmup_tensor = featuriser.t_radio(img)
    warmup_output = featuriser(warmup_tensor)
    warmup_ssl = warmup_output["backbone"][1]
    if pretrain: _ = cluster_model.few_shot_pretrain(warmup_ssl)

## Prompting
sim_threshold = 0.5
template_cls_token = featuriser.clip_global_feat_switch(img)
#pos_prompt_token = featuriser.clip_global_feat_switch("Printed Postcard ICRA Imperial Night Stempelkarte FHTW STAMP 2026 Vienna Red City")
#pos_prompt_token = featuriser.clip_global_feat_switch("a printed postcard of ICRA showing 2026 Vienna red city")
pos_prompt_token = featuriser.clip_global_feat_switch("a white printed paper card of ICRA 2026 Vienna showing a red city and grey stamp fields")
positive_feats = [template_cls_token, pos_prompt_token]

neg_prompts = ["background"]
if neg_prompts != [""]:
    negative_feats = [ featuriser.clip_global_feat_switch(prompt) for prompt in neg_prompts ]

@torch.inference_mode
def embed_img(pil_image: Image.Image) -> torch.Tensor:
    pil_image = pil_image.resize((img_size, img_size)) # This prevents huge pixel-level lang maps
    img_tensor = featuriser.t_radio(pil_image) # This auto-expands to batch size of 1
    radio_ret = featuriser(img_tensor)
    ssl_embeds = radio_ret["backbone"][1]
    cluster_maps = cluster_model(ssl_embeds)

    cluster_upscaled = F.interpolate(cluster_maps.float(), size=pil_image.size[::-1], mode="nearest").to(_target_dtype).int()
    pooled_embeds = masked_average_pooling(cluster_upscaled, radio_ret[featuriser.adaptor_name][1]) # radio patch idx 1 are patchtoken, 0 would be global token
    return pooled_embeds # Return as B, D, W, H

#############
## ROS 2 Node

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

        img = Image.fromarray(cv_image).convert("RGB")
        patch_token = embed_img(img)

        pos_sims = [ clip_similarity(patch_token, text_feats) for text_feats in positive_feats ]
        if neg_prompts != [""]:
            neg_sims = [ clip_similarity(patch_token, text_feats) for text_feats in negative_feats ]
        else: neg_sims = torch.zeros_like(pos_sims[0]) # This uses the same device as pos_sims

        lr_sims = sum(pos_sims)/len(pos_sims) - sum(neg_sims)/(len(neg_sims) + 1e-8) # Weight sum here to deal with len(pos) != len(neg)
        #lr_sims = clip_similarity(patch_token, template_cls_token)
        sim_min, sim_max = lr_sims.min(), lr_sims.max()
        sim_range = torch.clamp(sim_max - sim_min, min=0.2) # Prevent normalisation from blowing up noise due to low similarity
        lr_sims_norm = (lr_sims - sim_min) / (sim_range + 1e-8)

        # Upscale patch-level similarity to image size (H, W)
        sim_resized = F.interpolate(
            lr_sims_norm.unsqueeze(0).unsqueeze(0),
            size=img.size[::-1],  # (height, width)
            mode="nearest",
        ).squeeze().cpu().numpy()

        sim_resized[sim_resized < sim_threshold] = 0

        # Normalize similarity to [0, 1] for overlay
        #s_min, s_max = sim_resized.min(), sim_resized.max()
        #sim_norm = (sim_resized - s_min) / (s_max - s_min + 1e-8)
        #sim_norm[sim_norm < sim_threshold] = 0

        # Highlight high similarity in the red channel
        # With passthrough: rgb8 -> red=0, bgr8 -> red=2
        red_idx = 0 if msg.encoding == "rgb8" else 2
        red_strength = 1.0  # blend strength for the similarity overlay
        red_channel = cv_image[:, :, red_idx].astype(np.float32)
        red_channel = np.clip(
            red_channel + red_strength * sim_resized * 255, 0, 255
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
