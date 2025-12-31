import os
import sys
import json
import uuid
import time
import numpy as np
import torch
from typing import Optional, List, Dict, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

try:
    import folder_paths
    COMFY_MODELS_DIR = folder_paths.models_dir
    COMFY_OUTPUT_DIR = folder_paths.get_output_directory()
except ImportError:
    COMFY_MODELS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "models")
    COMFY_OUTPUT_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "output")

HYMOTION_MODELS_DIR = os.path.join(COMFY_MODELS_DIR, "HY-Motion")


def get_timestamp():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(t)) + f"{ms:03d}"

class HYMotionPipeline:
    def __init__(self, pipeline, device, config_path, fbx_converter=None):
        self.pipeline = pipeline
        self.device = device
        self.config_path = config_path
        self.fbx_converter = fbx_converter


class HYMotionData:
    def __init__(self, output_dict: Dict[str, Any], text: str, duration: float, seeds: List[int]):
        self.output_dict = output_dict
        self.text = text
        self.duration = duration
        self.seeds = seeds
        self.batch_size = output_dict["keypoints3d"].shape[0] if "keypoints3d" in output_dict else 1

class HYMotionLoadModel:
    @classmethod
    def INPUT_TYPES(s):
        model_options = ["HY-Motion-1.0", "HY-Motion-1.0-Lite"]
        quantization_options = ["none", "int8", "int4"]

        return {
            "required": {
                "model_name": (model_options, {"default": "HY-Motion-1.0-Lite"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "quantization": (quantization_options, {"default": "none"}),
            },
        }

    RETURN_TYPES = ("HYMOTION_PIPE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "load_model"
    CATEGORY = "HY-Motion"

    def load_model(self, model_name, device, quantization="none"):
        import yaml
        from .hymotion.utils.loaders import load_object

        model_dir = os.path.join(HYMOTION_MODELS_DIR, "ckpts", "tencent", model_name)
        config_path = os.path.join(model_dir, "config.yml")
        ckpt_path = os.path.join(model_dir, "latest.ckpt")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        print(f"[HY-Motion] Loading model config: {config_path}")
        print(f"[HY-Motion] Quantization: {quantization}")

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # Override quantization in text_encoder_cfg
        if "train_pipeline_args" in config and "text_encoder_cfg" in config["train_pipeline_args"]:
            config["train_pipeline_args"]["text_encoder_cfg"]["quantization"] = quantization if quantization != "none" else None

        pipeline = load_object(
            config["train_pipeline"],
            config["train_pipeline_args"],
            network_module=config["network_module"],
            network_module_args=config["network_module_args"],
        )

        allow_empty_ckpt = not os.path.exists(ckpt_path)
        if allow_empty_ckpt:
            print(f"[HY-Motion] Warning: Checkpoint not found at {ckpt_path}, using random initialized weights")

        model_dir = os.path.dirname(ckpt_path)
        stats_dir = os.path.join(model_dir, "stats")
        if not os.path.exists(stats_dir):
            plugin_stats = os.path.join(CURRENT_DIR, "HY-Motion-1.0", "stats")
            if os.path.exists(plugin_stats):
                stats_dir = plugin_stats
                print(f"[HY-Motion] Using stats from: {stats_dir}", flush=True)
            else:
                stats_dir = None
                print(f"[HY-Motion] WARNING: stats directory not found! Mean/Std will not be loaded.", flush=True)
        else:
            print(f"[HY-Motion] Using stats from: {stats_dir}", flush=True)

        pipeline.load_in_demo(
            ckpt_path,
            stats_dir,
            build_text_encoder=True,
            allow_empty_ckpt=allow_empty_ckpt,
        )

        target_device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        pipeline.to(target_device)
        pipeline.eval()

        print(f"[HY-Motion] Model loaded successfully, device: {target_device}", flush=True)

        import sys
        if hasattr(pipeline, 'mean') and hasattr(pipeline, 'std'):
            print(f"[HY-Motion] DEBUG: mean shape={pipeline.mean.shape}, mean range=[{pipeline.mean.min():.4f}, {pipeline.mean.max():.4f}]", flush=True)
            print(f"[HY-Motion] DEBUG: std shape={pipeline.std.shape}, std range=[{pipeline.std.min():.4f}, {pipeline.std.max():.4f}]", flush=True)
        else:
            print(f"[HY-Motion] WARNING: mean/std not found in pipeline!", flush=True)
        sys.stdout.flush()

        pipe_obj = HYMotionPipeline(
            pipeline=pipeline,
            device=target_device,
            config_path=config_path,
        )

        try:
            import fbx
            from .hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX
            plugin_dir = os.path.dirname(os.path.abspath(__file__))
            template_fbx_path = os.path.join(plugin_dir, "assets", "wooden_models", "boy_Rigging_smplx_tex.fbx")
            pipe_obj.fbx_converter = SMPLH2WoodFBX(template_fbx_path=template_fbx_path)
            print(f"[HY-Motion] FBX converter loaded")
        except ImportError:
            print(f"[HY-Motion] FBX SDK not installed, FBX export disabled")
        except Exception as e:
            print(f"[HY-Motion] Failed to load FBX converter: {e}")

        return (pipe_obj,)

class HYMotionGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HYMOTION_PIPE",),
                "text": ("STRING", {
                    "default": "A person is walking forward.",
                    "multiline": True,
                }),
                "duration": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.5,
                    "max": 12.0,
                    "step": 0.1,
                }),
                "seed": ("INT", {
                    "default": 42,
                    "min": 0,
                    "max": 0x7fffffff,
                }),
            },
            "optional": {
                "cfg_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 15.0,
                    "step": 0.5,
                }),
                "num_samples": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 4,
                }),
            }
        }

    RETURN_TYPES = ("HYMOTION_DATA",)
    RETURN_NAMES = ("motion_data",)
    FUNCTION = "generate"
    CATEGORY = "HY-Motion"

    def generate(self, pipeline: HYMotionPipeline, text: str, duration: float,
                 seed: int, cfg_scale: float = 5.0, num_samples: int = 1):
        import comfy.utils

        pipe = pipeline.pipeline

        seeds = [seed + i for i in range(num_samples)]

        print(f"[HY-Motion] Starting generation: text='{text[:50]}...', duration={duration}s, seeds={seeds}")

        # Create progress bar
        pbar = comfy.utils.ProgressBar(pipe.validation_steps)

        def progress_callback(current, total):
            pbar.update_absolute(current, total)

        with torch.no_grad():
            output_dict = pipe.generate(
                text=text,
                seed_input=seeds,
                duration_slider=duration,
                cfg_scale=cfg_scale,
                progress_callback=progress_callback,
            )

        import sys
        if "rot6d" in output_dict:
            rot6d = output_dict["rot6d"]
            print(f"[HY-Motion] DEBUG: rot6d shape={rot6d.shape}, range=[{rot6d.min():.4f}, {rot6d.max():.4f}]", flush=True)
        if "transl" in output_dict:
            transl = output_dict["transl"]
            print(f"[HY-Motion] DEBUG: transl shape={transl.shape}, range=[{transl.min():.4f}, {transl.max():.4f}]", flush=True)
        sys.stdout.flush()

        motion_data = HYMotionData(
            output_dict=output_dict,
            text=text,
            duration=duration,
            seeds=seeds,
        )

        print(f"[HY-Motion] Generation complete: batch_size={motion_data.batch_size}")

        return (motion_data,)

class HYMotionPreview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA",),
                "sample_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 3,
                }),
                "frame_step": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                }),
                "image_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("preview_frames",)
    FUNCTION = "render_preview"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def render_preview(self, motion_data: HYMotionData, sample_index: int = 0,
                       frame_step: int = 5, image_size: int = 512):
        import torch
        import numpy as np

        keypoints3d = motion_data.output_dict.get("keypoints3d")
        if keypoints3d is None:
            empty = torch.zeros(1, image_size, image_size, 3)
            return (empty,)

        if sample_index >= keypoints3d.shape[0]:
            sample_index = 0

        kpts = keypoints3d[sample_index].cpu().numpy()  # (L, J, 3)
        num_frames = kpts.shape[0]

        # Sample frames
        frame_indices = list(range(0, num_frames, frame_step))
        if len(frame_indices) == 0:
            frame_indices = [0]

        # Render each frame
        images = []
        for fi in frame_indices:
            frame_kpts = kpts[fi]  # (J, 3)
            img = self._render_skeleton_frame(frame_kpts, image_size)
            images.append(img)

        images_np = np.stack(images, axis=0)  # (B, H, W, C)
        images_tensor = torch.from_numpy(images_np).float() / 255.0

        return (images_tensor,)

    def _render_skeleton_frame(self, kpts: np.ndarray, size: int) -> np.ndarray:
        import numpy as np

        img = np.ones((size, size, 3), dtype=np.uint8) * 240

        # SMPL 22 joint bone connections (front view: X-Y coordinates)
        bones = [
            # Legs
            (0, 1), (1, 4), (4, 7), (7, 10),  # Left leg: Pelvis -> L_Hip -> L_Knee -> L_Ankle -> L_Foot
            (0, 2), (2, 5), (5, 8), (8, 11),  # Right leg: Pelvis -> R_Hip -> R_Knee -> R_Ankle -> R_Foot
            # Spine
            (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),  # Pelvis -> Spine1 -> Spine2 -> Spine3 -> Neck -> Head
            # Arms
            (9, 13), (13, 16), (16, 18), (18, 20),  # Left arm: Spine3 -> L_Collar -> L_Shoulder -> L_Elbow -> L_Wrist
            (9, 14), (14, 17), (17, 19), (19, 21),  # Right arm: Spine3 -> R_Collar -> R_Shoulder -> R_Elbow -> R_Wrist
        ]

        # Use X (horizontal) and Y (vertical) for front view
        x = kpts[:, 0]
        y = kpts[:, 1]

        margin = 0.15
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_range = max(x_max - x_min, 0.1)
        y_range = max(y_max - y_min, 0.1)
        scale = max(x_range, y_range) * (1 + 2 * margin)

        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2

        def to_pixel(px, py):
            px_norm = (px - cx) / scale + 0.5
            py_norm = (py - cy) / scale + 0.5
            return int(px_norm * size), int((1 - py_norm) * size)  # Flip Y for image coordinates

        # Draw bones with different colors for different body parts
        bone_colors = {
            'leg_l': (255, 100, 100),   # Red for left leg
            'leg_r': (100, 100, 255),   # Blue for right leg
            'spine': (100, 200, 100),   # Green for spine
            'arm_l': (255, 150, 50),    # Orange for left arm
            'arm_r': (50, 150, 255),    # Cyan for right arm
        }

        for i, (b1, b2) in enumerate(bones):
            if b1 < len(kpts) and b2 < len(kpts):
                p1 = to_pixel(x[b1], y[b1])
                p2 = to_pixel(x[b2], y[b2])
                if i < 4:
                    color = bone_colors['leg_l']
                elif i < 8:
                    color = bone_colors['leg_r']
                elif i < 13:
                    color = bone_colors['spine']
                elif i < 17:
                    color = bone_colors['arm_l']
                else:
                    color = bone_colors['arm_r']
                self._draw_line(img, p1, p2, color, 3)

        # Draw joints
        for i in range(min(22, len(kpts))):
            px, py = to_pixel(x[i], y[i])
            if 0 <= px < size and 0 <= py < size:
                radius = 5
                for dx in range(-radius, radius + 1):
                    for dy in range(-radius, radius + 1):
                        if dx*dx + dy*dy <= radius*radius:
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < size and 0 <= ny < size:
                                img[ny, nx] = [50, 50, 50]

        return img

    def _draw_line(self, img: np.ndarray, p1: tuple, p2: tuple, color: tuple, thickness: int):
        x1, y1 = p1
        x2, y2 = p2
        h, w = img.shape[:2]

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        steps = max(dx, dy, 1)

        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))

            for tx in range(-thickness, thickness + 1):
                for ty in range(-thickness, thickness + 1):
                    nx, ny = x + tx, y + ty
                    if 0 <= nx < w and 0 <= ny < h:
                        img[ny, nx] = color


class HYMotionExportFBX:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HYMOTION_PIPE",),
                "motion_data": ("HYMOTION_DATA",),
                "output_dir": ("STRING", {
                    "default": "hymotion_fbx",
                }),
                "filename_prefix": ("STRING", {
                    "default": "motion",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("fbx_paths",)
    FUNCTION = "export_fbx"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def export_fbx(self, pipeline: HYMotionPipeline, motion_data: HYMotionData,
                   output_dir: str, filename_prefix: str):

        if pipeline.fbx_converter is None:
            return ("FBX SDK not installed, export unavailable",)

        # Use ComfyUI's native output directory
        full_output_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(full_output_dir, exist_ok=True)

        output_dict = motion_data.output_dict
        timestamp = get_timestamp()
        unique_id = str(uuid.uuid4())[:8]

        fbx_files = []

        from .hymotion.pipeline.body_model import construct_smpl_data_dict

        import sys
        for batch_idx in range(motion_data.batch_size):
            try:
                rot6d = output_dict["rot6d"][batch_idx].clone()
                transl = output_dict["transl"][batch_idx].clone()
                print(f"[HY-Motion] batch {batch_idx}: rot6d shape={rot6d.shape}, transl shape={transl.shape}", flush=True)
                print(f"[HY-Motion] batch {batch_idx}: rot6d range=[{rot6d.min():.4f}, {rot6d.max():.4f}]", flush=True)
                print(f"[HY-Motion] batch {batch_idx}: transl range=[{transl.min():.4f}, {transl.max():.4f}]", flush=True)
                smpl_data = construct_smpl_data_dict(rot6d, transl)
                print(f"[HY-Motion] construct_smpl_data_dict success", flush=True)
            except Exception as e:
                import traceback
                print(f"[HY-Motion] construct_smpl_data_dict failed: {e}", flush=True)
                traceback.print_exc()
                sys.stdout.flush()
                continue

            fbx_filename = f"{filename_prefix}_{timestamp}_{unique_id}_{batch_idx:03d}.fbx"
            fbx_path = os.path.join(full_output_dir, fbx_filename)

            try:
                print(f"[HY-Motion] Converting to FBX: {fbx_path}", flush=True)
                print(f"[HY-Motion] smpl_data keys: {smpl_data.keys()}", flush=True)
                poses = smpl_data.get('poses')
                trans = smpl_data.get('trans')
                if poses is not None:
                    print(f"[HY-Motion] poses shape: {poses.shape}, range=[{poses.min():.4f}, {poses.max():.4f}]", flush=True)
                if trans is not None:
                    print(f"[HY-Motion] trans shape: {trans.shape}, range=[{trans.min():.4f}, {trans.max():.4f}]", flush=True)
                sys.stdout.flush()
                success = pipeline.fbx_converter.convert_npz_to_fbx(smpl_data, fbx_path)
                print(f"[HY-Motion] convert_npz_to_fbx returned: {success}")
                if success:
                    fbx_files.append(fbx_path)
                    print(f"[HY-Motion] FBX export successful: {fbx_path}")

                    txt_path = fbx_path.replace(".fbx", ".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(motion_data.text)
                else:
                    print(f"[HY-Motion] FBX export returned False")
            except Exception as e:
                import traceback
                print(f"[HY-Motion] FBX export failed: {e}")
                traceback.print_exc()

        # Return paths relative to ComfyUI output directory
        relative_paths = [os.path.relpath(p, COMFY_OUTPUT_DIR) for p in fbx_files]
        result = "\n".join(relative_paths) if relative_paths else "Export failed"
        return (result,)


class HYMotionSaveNPZ:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "motion_data": ("HYMOTION_DATA",),
                "output_dir": ("STRING", {
                    "default": "hymotion_npz",
                }),
                "filename_prefix": ("STRING", {
                    "default": "motion",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("npz_paths",)
    FUNCTION = "save_npz"
    CATEGORY = "HY-Motion"
    OUTPUT_NODE = True

    def save_npz(self, motion_data: HYMotionData, output_dir: str, filename_prefix: str):
        import numpy as np

        # Use ComfyUI's native output directory
        full_output_dir = os.path.join(COMFY_OUTPUT_DIR, output_dir)
        os.makedirs(full_output_dir, exist_ok=True)

        output_dict = motion_data.output_dict
        timestamp = get_timestamp()
        unique_id = str(uuid.uuid4())[:8]

        npz_files = []

        for batch_idx in range(motion_data.batch_size):
            data = {}
            for key in ["keypoints3d", "rot6d", "transl", "root_rotations_mat"]:
                if key in output_dict and output_dict[key] is not None:
                    tensor = output_dict[key][batch_idx]
                    if hasattr(tensor, 'cpu'):
                        data[key] = tensor.cpu().numpy()
                    else:
                        data[key] = np.array(tensor)

            data["text"] = motion_data.text
            data["duration"] = motion_data.duration
            data["seed"] = motion_data.seeds[batch_idx] if batch_idx < len(motion_data.seeds) else 0

            npz_filename = f"{filename_prefix}_{timestamp}_{unique_id}_{batch_idx:03d}.npz"
            npz_path = os.path.join(full_output_dir, npz_filename)

            np.savez(npz_path, **data)
            npz_files.append(npz_path)
            print(f"[HY-Motion] NPZ saved: {npz_path}")

        # Return paths relative to ComfyUI output directory
        relative_paths = [os.path.relpath(p, COMFY_OUTPUT_DIR) for p in npz_files]
        result = "\n".join(relative_paths)
        return (result,)


NODE_CLASS_MAPPINGS = {
    "HYMotionLoadModel": HYMotionLoadModel,
    "HYMotionGenerate": HYMotionGenerate,
    "HYMotionPreview": HYMotionPreview,
    "HYMotionExportFBX": HYMotionExportFBX,
    "HYMotionSaveNPZ": HYMotionSaveNPZ,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HYMotionLoadModel": "HY-Motion Load Model",
    "HYMotionGenerate": "HY-Motion Generate",
    "HYMotionPreview": "HY-Motion Preview",
    "HYMotionExportFBX": "HY-Motion Export FBX",
    "HYMotionSaveNPZ": "HY-Motion Save NPZ",
}
