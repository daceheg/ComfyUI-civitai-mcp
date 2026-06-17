"""
Metadata-aware image loaders for the Civitai post pipeline.

ComfyUI's stock ``Load Image`` outputs only an IMAGE/MASK tensor and discards the
embedded generation metadata (the PNG ``parameters`` text chunk / JPEG EXIF
UserComment, plus the ``prompt``/``workflow`` JSON chunks). These loaders read the
file directly so they can recover that metadata and hand it to the Civitai post
nodes:

  * ``a1111_params`` -- the A1111/Civitai parameters string, ready to wire into
    Civitai Post Image / Create Post. For stock-SaveImage files that carry only a
    ``prompt`` graph (no parameters string), one is synthesized via
    ``auto_build_params``.
  * ``workflow_json`` -- the embedded ComfyUI workflow, passed through so the post
    nodes can re-embed it (wired input wins over the live graph).

Pair these with ``Civitai Parse A1111 Params`` to break the string into individual
typed fields when needed.
"""

import os
import json
import hashlib

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

try:
    import folder_paths  # provided by ComfyUI at runtime
except Exception:  # pragma: no cover - allows host-side unit testing
    folder_paths = None

from . import civitai_metadata


def _pil_to_image_and_mask(img):
    """Convert a PIL image (possibly multi-frame) to (IMAGE, MASK) tensors.

    Mirrors ComfyUI's stock LoadImage: EXIF-transpose each frame, stack same-size
    frames into a batch, and derive the mask from the alpha channel (inverted) or
    a zero mask when there's no alpha.
    """
    output_images = []
    output_masks = []
    w, h = None, None

    for frame in ImageSequence.Iterator(img):
        frame = ImageOps.exif_transpose(frame)
        rgb = frame.convert("RGB")

        if len(output_images) == 0:
            w, h = rgb.size

        if rgb.size[0] != w or rgb.size[1] != h:
            continue

        arr = np.array(rgb).astype(np.float32) / 255.0
        output_images.append(torch.from_numpy(arr)[None,])

        if "A" in frame.getbands():
            mask_arr = np.array(frame.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask_arr)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32)
        output_masks.append(mask.unsqueeze(0))

    image = torch.cat(output_images, dim=0)
    mask = torch.cat(output_masks, dim=0)
    return image, mask


def _extract_for_path(image_path):
    """Open an image file and return (image_tensor, mask_tensor, meta_dict).

    ``meta_dict`` is the result of ``extract_embedded_metadata``. The image is
    re-opened for tensor conversion because the metadata read consumes the file
    object's frame iterator.
    """
    with Image.open(image_path) as img:
        meta = civitai_metadata.extract_embedded_metadata(img)
        image, mask = _pil_to_image_and_mask(img)
    return image, mask, meta


def _workflow_json_str(meta):
    """Serialize the embedded workflow (or prompt graph as fallback) to a JSON
    string for the workflow_json output, or "" when neither is present."""
    workflow = meta.get("workflow") or meta.get("prompt_graph")
    if isinstance(workflow, dict) and workflow:
        return json.dumps(workflow)
    return ""


class CivitaiLoadImageWithMetadata:
    """LoadImage-style node that also recovers embedded generation metadata."""

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory() if folder_paths else "."
        try:
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        except Exception:
            files = []
        if folder_paths:
            files = folder_paths.filter_files_content_types(files, ["image"])
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "a1111_params", "workflow_json")
    FUNCTION = "load_image"
    CATEGORY = "Civitai-mcp"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image) if folder_paths else image
        img_tensor, mask, meta = _extract_for_path(image_path)

        params = meta.get("a1111_params", "")
        workflow_json = _workflow_json_str(meta)

        status = "with metadata" if params else "no embedded metadata"
        print(f"[Civitai MCP] Loaded '{os.path.basename(image_path)}' ({status}).")
        return (img_tensor, mask, params, workflow_json)

    @classmethod
    def IS_CHANGED(s, image):
        if not folder_paths:
            return image
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, "rb") as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if folder_paths and not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True


# Image extensions we attempt to read metadata from, in directory order.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff")


class CivitaiLoadImagesFromDir:
    """Load every image in a directory, each with its embedded metadata.

    Returns parallel lists (IMAGE / a1111_params / filename) so a downstream
    ``Civitai Create Post`` (INPUT_IS_LIST) can post the whole batch with
    per-image metadata. ``image_load_cap`` and ``start_index`` page through large
    directories; files are taken in sorted name order.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
            },
            "optional": {
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "tooltip": "Maximum number of images to load (0 = no limit)."}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 0xffffffff, "tooltip": "Skip this many images (sorted by filename) before loading."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "a1111_params", "workflow_json", "filenames")
    OUTPUT_IS_LIST = (True, True, True, True)
    FUNCTION = "load_images"
    CATEGORY = "Civitai-mcp"

    def load_images(self, directory, image_load_cap=0, start_index=0):
        if not directory or not os.path.isdir(directory):
            raise ValueError(f"Directory not found: {directory!r}")

        names = sorted(
            f for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        )
        names = names[start_index:]
        if image_load_cap > 0:
            names = names[:image_load_cap]

        if not names:
            raise ValueError(f"No images found in {directory!r} (start_index={start_index}).")

        images, params_list, workflow_list, filenames = [], [], [], []
        for name in names:
            path = os.path.join(directory, name)
            try:
                img_tensor, _mask, meta = _extract_for_path(path)
            except Exception as e:
                print(f"[Civitai MCP] Warning: skipped '{name}': {e}")
                continue
            images.append(img_tensor)
            params_list.append(meta.get("a1111_params", ""))
            workflow_list.append(_workflow_json_str(meta))
            filenames.append(name)

        if not images:
            raise ValueError(f"No loadable images in {directory!r}.")

        with_meta = sum(1 for p in params_list if p)
        print(f"[Civitai MCP] Loaded {len(images)} image(s) from '{directory}' ({with_meta} with metadata).")
        return (images, params_list, workflow_list, filenames)


class CivitaiParseA1111Params:
    """Break an A1111/Civitai parameters string into individual typed fields.

    The inverse of the string the post nodes embed. Wire it from a loader's
    ``a1111_params`` output (or paste a string) to drive other nodes with the
    prompt, seed, sampler, dimensions, etc.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "a1111_params": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "negative_prompt", "seed", "steps", "cfg_scale", "sampler_name", "width", "height")
    FUNCTION = "parse"
    CATEGORY = "Civitai-mcp"

    def parse(self, a1111_params):
        f = civitai_metadata.parse_a1111_params(a1111_params)
        print("[Civitai MCP] Parsed A1111 parameters.")
        return (
            f["prompt"], f["negative_prompt"], f["seed"], f["steps"],
            f["cfg_scale"], f["sampler_name"], f["width"], f["height"],
        )
