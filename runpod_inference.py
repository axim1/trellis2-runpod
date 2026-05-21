import base64
import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Normalize Hugging Face auth env vars so local tests and serverless deployments
# can provide either name without changing the code.
if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
    os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
elif os.environ.get("HUGGING_FACE_HUB_TOKEN") and not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]

import cv2
import numpy as np
import requests
import torch
import boto3
from PIL import Image

import o_voxel
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.renderers import EnvMap
from trellis2.utils import render_utils


ROOT_DIR = Path(__file__).resolve().parent
MODEL_ID = "microsoft/TRELLIS.2-4B"
MAX_SEED = np.iinfo(np.int32).max
PIPELINE_TYPES = {
    "512": "512",
    "1024": "1024_cascade",
    "1536": "1536_cascade",
}
PREVIEW_KEYS = {
    "normal",
    "clay",
    "base_color",
    "shaded_forest",
    "shaded_sunset",
    "shaded_courtyard",
}
DEFAULTS = {
    "resolution": "1024",
    "decimation_target": 500000,
    "texture_size": 2048,
    "ss_guidance_strength": 7.5,
    "ss_guidance_rescale": 0.7,
    "ss_sampling_steps": 12,
    "ss_rescale_t": 5.0,
    "shape_slat_guidance_strength": 7.5,
    "shape_slat_guidance_rescale": 0.5,
    "shape_slat_sampling_steps": 12,
    "shape_slat_rescale_t": 3.0,
    "tex_slat_guidance_strength": 1.0,
    "tex_slat_guidance_rescale": 0.0,
    "tex_slat_sampling_steps": 12,
    "tex_slat_rescale_t": 3.0,
    "preview_resolution": 512,
    "preview_mode": "shaded_forest",
}


def _to_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _to_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _to_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _normalize_base64(data: str) -> str:
    if "," in data and data.split(",", 1)[0].startswith("data:"):
        return data.split(",", 1)[1]
    return data


def decode_base64_image(data: str) -> Image.Image:
    raw = base64.b64decode(_normalize_base64(data))
    return Image.open(io.BytesIO(raw)).convert("RGBA")


def load_image_from_url(url: str, timeout: int = 30) -> Image.Image:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGBA")


def image_to_data_url(image: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    buffer = io.BytesIO()
    save_image = image
    if fmt.upper() == "JPEG":
        save_image = image.convert("RGB")
        save_image.save(buffer, format=fmt, quality=quality)
        media_type = "image/jpeg"
    else:
        save_image.save(buffer, format=fmt)
        media_type = f"image/{fmt.lower()}"
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:{media_type};base64,{encoded}"


def file_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def r2_is_configured() -> bool:
    return bool(
        os.environ.get("R2_BUCKET")
        and os.environ.get("R2_ACCESS_KEY_ID")
        and os.environ.get("R2_SECRET_ACCESS_KEY")
        and (os.environ.get("R2_ENDPOINT") or os.environ.get("R2_ACCOUNT_ID"))
    )


def upload_file_to_r2(path: Path, content_type: str = "model/gltf-binary") -> Optional[Dict[str, str]]:
    if not r2_is_configured():
        return None

    bucket = os.environ["R2_BUCKET"]
    endpoint = os.environ.get("R2_ENDPOINT")
    if not endpoint:
        endpoint = f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com"

    key_prefix = os.environ.get("R2_KEY_PREFIX", "trellis-models").strip("/")
    object_key = f"{key_prefix}/{path.name}" if key_prefix else path.name

    client = boto3.client(
        "s3",
        region_name="auto",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    )
    client.upload_file(
        str(path),
        bucket,
        object_key,
        ExtraArgs={"ContentType": content_type},
    )

    result: Dict[str, str] = {
        "glb_storage": "r2",
        "glb_object_key": object_key,
        "glb_bucket": bucket,
    }
    public_base = os.environ.get("R2_PUBLIC_BASE_URL")
    if public_base:
        result["glb_url"] = f"{public_base.rstrip('/')}/{object_key}"
    return result


def load_input_image(input_data: Dict[str, Any]) -> Image.Image:
    if input_data.get("image_base64"):
        return decode_base64_image(input_data["image_base64"])

    if input_data.get("image_url"):
        return load_image_from_url(input_data["image_url"])

    image_value = input_data.get("image")
    if isinstance(image_value, str) and image_value.strip():
        if image_value.startswith(("http://", "https://")):
            return load_image_from_url(image_value)
        return decode_base64_image(image_value)

    raise ValueError("Input must include 'image_base64', 'image_url', or 'image'.")


def resolve_preview_modes(preview_mode: Any) -> List[str]:
    if preview_mode is None or preview_mode is False:
        return []

    if isinstance(preview_mode, str):
        if preview_mode == "all":
            return sorted(PREVIEW_KEYS)
        if preview_mode not in PREVIEW_KEYS:
            raise ValueError(
                f"Invalid preview_mode '{preview_mode}'. Expected one of {sorted(PREVIEW_KEYS)} or 'all'."
            )
        return [preview_mode]

    if isinstance(preview_mode, Iterable):
        modes = list(preview_mode)
        invalid = [mode for mode in modes if mode not in PREVIEW_KEYS]
        if invalid:
            raise ValueError(
                f"Invalid preview_mode entries {invalid}. Expected values from {sorted(PREVIEW_KEYS)}."
            )
        return modes

    raise ValueError("preview_mode must be a string, a list of preview keys, or 'all'.")


@dataclass
class GenerationOptions:
    seed: int
    resolution: str
    decimation_target: int
    texture_size: int
    preprocess_image: bool
    include_glb_base64: bool
    preview_resolution: int
    preview_modes: List[str]
    ss_guidance_strength: float
    ss_guidance_rescale: float
    ss_sampling_steps: int
    ss_rescale_t: float
    shape_slat_guidance_strength: float
    shape_slat_guidance_rescale: float
    shape_slat_sampling_steps: int
    shape_slat_rescale_t: float
    tex_slat_guidance_strength: float
    tex_slat_guidance_rescale: float
    tex_slat_sampling_steps: int
    tex_slat_rescale_t: float

    @classmethod
    def from_input(cls, input_data: Dict[str, Any]) -> "GenerationOptions":
        resolution = str(input_data.get("resolution", DEFAULTS["resolution"]))
        if resolution not in PIPELINE_TYPES:
            raise ValueError(f"Invalid resolution '{resolution}'. Expected one of {sorted(PIPELINE_TYPES)}.")

        if _to_bool(input_data.get("randomize_seed"), False):
            seed = int(np.random.randint(0, MAX_SEED))
        else:
            seed = _to_int(input_data.get("seed"), 0)

        include_preview = _to_bool(input_data.get("include_preview"), True)
        preview_modes = []
        if include_preview:
            preview_modes = resolve_preview_modes(input_data.get("preview_mode", DEFAULTS["preview_mode"]))

        return cls(
            seed=seed,
            resolution=resolution,
            decimation_target=_to_int(input_data.get("decimation_target"), DEFAULTS["decimation_target"]),
            texture_size=_to_int(input_data.get("texture_size"), DEFAULTS["texture_size"]),
            preprocess_image=_to_bool(input_data.get("preprocess_image"), True),
            include_glb_base64=_to_bool(input_data.get("include_glb_base64"), False),
            preview_resolution=_to_int(input_data.get("preview_resolution"), DEFAULTS["preview_resolution"]),
            preview_modes=preview_modes,
            ss_guidance_strength=_to_float(
                input_data.get("ss_guidance_strength"),
                DEFAULTS["ss_guidance_strength"],
            ),
            ss_guidance_rescale=_to_float(
                input_data.get("ss_guidance_rescale"),
                DEFAULTS["ss_guidance_rescale"],
            ),
            ss_sampling_steps=_to_int(
                input_data.get("ss_sampling_steps"),
                DEFAULTS["ss_sampling_steps"],
            ),
            ss_rescale_t=_to_float(
                input_data.get("ss_rescale_t"),
                DEFAULTS["ss_rescale_t"],
            ),
            shape_slat_guidance_strength=_to_float(
                input_data.get("shape_slat_guidance_strength"),
                DEFAULTS["shape_slat_guidance_strength"],
            ),
            shape_slat_guidance_rescale=_to_float(
                input_data.get("shape_slat_guidance_rescale"),
                DEFAULTS["shape_slat_guidance_rescale"],
            ),
            shape_slat_sampling_steps=_to_int(
                input_data.get("shape_slat_sampling_steps"),
                DEFAULTS["shape_slat_sampling_steps"],
            ),
            shape_slat_rescale_t=_to_float(
                input_data.get("shape_slat_rescale_t"),
                DEFAULTS["shape_slat_rescale_t"],
            ),
            tex_slat_guidance_strength=_to_float(
                input_data.get("tex_slat_guidance_strength"),
                DEFAULTS["tex_slat_guidance_strength"],
            ),
            tex_slat_guidance_rescale=_to_float(
                input_data.get("tex_slat_guidance_rescale"),
                DEFAULTS["tex_slat_guidance_rescale"],
            ),
            tex_slat_sampling_steps=_to_int(
                input_data.get("tex_slat_sampling_steps"),
                DEFAULTS["tex_slat_sampling_steps"],
            ),
            tex_slat_rescale_t=_to_float(
                input_data.get("tex_slat_rescale_t"),
                DEFAULTS["tex_slat_rescale_t"],
            ),
        )


class TrellisRunpodRuntime:
    def __init__(self) -> None:
        self.pipeline: Optional[Trellis2ImageTo3DPipeline] = None
        self.envmaps: Dict[str, EnvMap] = {}
        self.model_load_seconds: Optional[float] = None

    def load(self) -> None:
        if self.pipeline is not None:
            return

        start = time.perf_counter()
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
        self.pipeline.cuda()
        self.envmaps = {
            "forest": self._load_envmap("forest.exr"),
            "sunset": self._load_envmap("sunset.exr"),
            "courtyard": self._load_envmap("courtyard.exr"),
        }
        self.model_load_seconds = time.perf_counter() - start

    def _load_envmap(self, filename: str) -> EnvMap:
        envmap_path = ROOT_DIR / "assets" / "hdri" / filename
        image = cv2.cvtColor(cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)
        return EnvMap(torch.tensor(image, dtype=torch.float32, device="cuda"))

    def generate(self, image: Image.Image, options: GenerationOptions) -> Dict[str, Any]:
        was_loaded = self.pipeline is not None
        self.load()
        assert self.pipeline is not None

        output_dir = Path("/tmp/trellis2")
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / f"trellis2_{int(time.time() * 1000)}_{options.seed}.glb"

        start = time.perf_counter()
        try:
            with torch.inference_mode():
                # Match the official TRELLIS app flow as closely as possible:
                # preprocess the uploaded image first, then run the pipeline with
                # preprocess_image disabled so generation operates on the already
                # prepared foreground crop.
                processed_image = image
                if options.preprocess_image:
                    processed_image = self.pipeline.preprocess_image(image)

                meshes = self.pipeline.run(
                    processed_image,
                    seed=options.seed,
                    preprocess_image=False,
                    sparse_structure_sampler_params={
                        "steps": options.ss_sampling_steps,
                        "guidance_strength": options.ss_guidance_strength,
                        "guidance_rescale": options.ss_guidance_rescale,
                        "rescale_t": options.ss_rescale_t,
                    },
                    shape_slat_sampler_params={
                        "steps": options.shape_slat_sampling_steps,
                        "guidance_strength": options.shape_slat_guidance_strength,
                        "guidance_rescale": options.shape_slat_guidance_rescale,
                        "rescale_t": options.shape_slat_rescale_t,
                    },
                    tex_slat_sampler_params={
                        "steps": options.tex_slat_sampling_steps,
                        "guidance_strength": options.tex_slat_guidance_strength,
                        "guidance_rescale": options.tex_slat_guidance_rescale,
                        "rescale_t": options.tex_slat_rescale_t,
                    },
                    pipeline_type=PIPELINE_TYPES[options.resolution],
                )
                mesh = meshes[0]
                mesh.simplify(16777216)

                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=mesh.layout,
                    voxel_size=mesh.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=options.decimation_target,
                    texture_size=options.texture_size,
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    verbose=True,
                )
                glb.export(str(artifact_path), extension_webp=True)

                preview_images: Dict[str, str] = {}
                if options.preview_modes:
                    snapshots = render_utils.render_snapshot(
                        mesh,
                        resolution=options.preview_resolution,
                        r=2,
                        fov=36,
                        nviews=1,
                        envmap=self.envmaps,
                    )
                    for mode in options.preview_modes:
                        preview_images[mode] = image_to_data_url(Image.fromarray(snapshots[mode][0]))
        finally:
            torch.cuda.empty_cache()

        result: Dict[str, Any] = {
            "seed": options.seed,
            "resolution": options.resolution,
            "decimation_target": options.decimation_target,
            "texture_size": options.texture_size,
            "cold_start": not was_loaded,
            "model_load_seconds": self.model_load_seconds if not was_loaded else 0.0,
            "generation_seconds": time.perf_counter() - start,
            "glb_filename": artifact_path.name,
            "glb_size_bytes": artifact_path.stat().st_size,
        }

        uploaded_glb = upload_file_to_r2(artifact_path)
        if uploaded_glb:
            result.update(uploaded_glb)

        if options.include_glb_base64:
            result["glb_base64"] = file_to_base64(artifact_path)

        if options.preview_modes:
            result["preview_images"] = preview_images

        return result
