import io
import json
import base64
import torch
from PIL import Image
from . import civitai_api
from . import civitai_metadata


def _tensor_to_pil(image_tensor):
    """Convert a single ComfyUI image tensor [H, W, C] to a PIL image."""
    img_np = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(img_np)


def _parse_workflow_json(workflow_json):
    """Parse a workflow_json string (from a Civitai loader) into a graph dict.

    Returns the dict, or None when empty/unparseable -- in which case the post
    nodes fall back to the live workflow from the hidden prompt/extra_pnginfo.
    """
    if isinstance(workflow_json, dict):
        return workflow_json or None
    if isinstance(workflow_json, str) and workflow_json.strip():
        try:
            parsed = json.loads(workflow_json)
            return parsed if isinstance(parsed, dict) and parsed else None
        except Exception:
            print("[Civitai MCP] Warning: workflow_json could not be parsed; using live workflow.")
    return None

class CivitaiPostImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "publish": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "title": ("STRING", {"default": "", "multiline": False}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "model_version_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "collection_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "a1111_params": ("STRING", {"default": "", "multiline": True, "tooltip": "A1111-format generation parameters to embed. Wire ComfyUI-Image-Saver's 'a1111_params' output here for full Civitai metadata (prompt, sampler, Civitai resources/AIRs). If empty, the node auto-detects from the workflow."}),
                "workflow_json": ("STRING", {"default": "", "multiline": True, "forceInput": True, "tooltip": "ComfyUI workflow JSON to embed (PNG only). Wire a Civitai metadata loader's 'workflow_json' output here to preserve the original graph of a re-posted image. Takes precedence over the live workflow; falls back to it when empty."}),
                "embed_metadata": ("BOOLEAN", {"default": True, "tooltip": "Embed generation parameters into the uploaded image so Civitai shows full metadata."}),
                "embed_workflow": ("BOOLEAN", {"default": True, "tooltip": "Embed the ComfyUI workflow + prompt graph (PNG only) so the post is reproducible by drag-and-drop into ComfyUI."}),
                "file_format": (["png", "jpg"], {"default": "png", "tooltip": "PNG carries metadata + workflow in text chunks (lossless). JPG carries the parameters in EXIF (smaller, no workflow)."}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "tooltip": "JPEG quality (only used when file_format is jpg)."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    OUTPUT_NODE = True

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("post_id", "post_url")
    FUNCTION = "post_image"
    CATEGORY = "Civitai-mcp"

    def post_image(self, image, publish, title="", description="", model_version_id=0,
                   collection_id=0, a1111_params="", workflow_json="", embed_metadata=True,
                   embed_workflow=True, file_format="png", jpg_quality=95, prompt=None,
                   extra_pnginfo=None):
        # ComfyUI images are tensors with shape [B, H, W, C]
        # We only take the first image if it's a batch
        if len(image.shape) == 4 and image.shape[0] > 1:
            print(f"[Civitai MCP] Warning: 'Civitai Post Image' received a batch of {image.shape[0]} images. Only the first image will be posted. Use 'Civitai Create Post' for multiple images.")

        single_img = image[0]
        pil_img = _tensor_to_pil(single_img)
        width, height = pil_img.size

        # Resolve the parameters string: explicit input wins, else auto-detect.
        params_str = a1111_params.strip() if isinstance(a1111_params, str) else ""
        if embed_metadata and not params_str:
            params_str = civitai_metadata.auto_build_params(prompt, width=width, height=height)
            if params_str:
                print("[Civitai MCP] Auto-detected generation metadata from the workflow.")

        workflow_override = _parse_workflow_json(workflow_json)
        img_bytes, content_type = civitai_metadata.encode_image(
            pil_img, file_format=file_format, a1111_params=params_str,
            embed_metadata=embed_metadata, embed_workflow=embed_workflow,
            prompt=prompt, extra_pnginfo=extra_pnginfo, jpg_quality=jpg_quality,
            workflow_override=workflow_override,
        )
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        print(f"[Civitai MCP] Uploading single image ({width}x{height}, {content_type})...")

        # Upload image to Civitai via MCP
        upload_res = civitai_api.upload_image(img_base64, content_type=content_type)
        uuid = upload_res.get("uuid")
        if not uuid:
            raise Exception("Failed to upload image: UUID was not returned.")
            
        print(f"[Civitai MCP] Image uploaded. UUID: {uuid}")
        
        post_images = [{
            "uuid": uuid,
            "width": width,
            "height": height,
            "type": "image"
        }]
        
        final_title = title.strip() if title.strip() else None
        final_desc = description.strip() if description.strip() else None
        
        print("[Civitai MCP] Creating post...")
        post_res = civitai_api.create_post(
            images=post_images,
            title=final_title,
            detail=final_desc,
            publish=publish,
            model_version_id=model_version_id if model_version_id > 0 else None,
            collection_id=collection_id if collection_id > 0 else None
        )
        
        post_id = post_res.get("id", 0)
        post_url = post_res.get("url", f"https://civitai.com/posts/{post_id}" if post_id else "")
        
        print(f"[Civitai MCP] Post created successfully! URL: {post_url}")
        
        return (post_id, post_url)


class CivitaiCreatePost:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "publish": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "title": ("STRING", {"default": "", "multiline": False}),
                "description": ("STRING", {"default": "", "multiline": True}),
                "max_images_per_post": ("INT", {"default": 20, "min": 1, "max": 20, "tooltip": "Civitai allows at most 20 images per post. If more images are passed, they are split across multiple posts of up to this many images each."}),
                "model_version_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "collection_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "a1111_params": ("STRING", {"default": "", "multiline": True, "tooltip": "A1111-format generation parameters to embed. Wire ComfyUI-Image-Saver's 'a1111_params' output here for full Civitai metadata. Accepts a list (one per image); a single value is applied to all images. If empty, auto-detected from the workflow."}),
                "workflow_json": ("STRING", {"default": "", "multiline": True, "forceInput": True, "tooltip": "ComfyUI workflow JSON to embed (PNG only). Wire a Civitai metadata loader's 'workflow_json' output here to preserve the original graph of re-posted images. Takes precedence over the live workflow; falls back to it when empty."}),
                "embed_metadata": ("BOOLEAN", {"default": True, "tooltip": "Embed generation parameters into each uploaded image so Civitai shows full metadata."}),
                "embed_workflow": ("BOOLEAN", {"default": True, "tooltip": "Embed the ComfyUI workflow + prompt graph (PNG only) so the post is reproducible by drag-and-drop into ComfyUI."}),
                "file_format": (["png", "jpg"], {"default": "png", "tooltip": "PNG carries metadata + workflow in text chunks (lossless). JPG carries the parameters in EXIF (smaller, no workflow)."}),
                "jpg_quality": ("INT", {"default": 95, "min": 1, "max": 100, "tooltip": "JPEG quality (only used when file_format is jpg)."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    INPUT_IS_LIST = True

    OUTPUT_NODE = True

    RETURN_TYPES = ("INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("post_id", "post_url", "post_ids_text", "post_urls_text")
    # post_id/post_url are lists (one entry per post created); post_ids_text and
    # post_urls_text are single newline-joined strings so any display/save node
    # (which only sees the first element of a list input) shows every post.
    OUTPUT_IS_LIST = (True, True, False, False)
    FUNCTION = "create_post"
    CATEGORY = "Civitai-mcp"

    @staticmethod
    def _first(value, default=None):
        """INPUT_IS_LIST delivers every input as a list; take the first scalar."""
        return value[0] if isinstance(value, list) and len(value) > 0 else default

    @staticmethod
    def _per_image(values, idx, default=""):
        """Resolve a per-image value from a string list for image ``idx``.

        A single-element list is broadcast to every image; otherwise the entry at
        ``idx`` is used, falling back to ``default`` when there are more images
        than entries.
        """
        if not values:
            return default
        if len(values) == 1:
            return values[0]
        return values[idx] if idx < len(values) else default

    def create_post(self, images, publish, title="", description="", max_images_per_post=20,
                    model_version_id=0, collection_id=0, a1111_params="", workflow_json="",
                    embed_metadata=True, embed_workflow=True, file_format="png", jpg_quality=95,
                    prompt=None, extra_pnginfo=None):
        # With INPUT_IS_LIST, every argument is a list. Scalars take the first item.
        single_publish = self._first(publish, True)
        single_title = self._first(title, "")
        single_desc = self._first(description, "")
        single_max_per_post = self._first(max_images_per_post, 20)
        single_model_version = self._first(model_version_id, 0)
        single_collection = self._first(collection_id, 0)
        single_embed_metadata = self._first(embed_metadata, True)
        single_embed_workflow = self._first(embed_workflow, True)
        single_format = self._first(file_format, "png")
        single_quality = self._first(jpg_quality, 95)
        # Live workflow/prompt graph is shared across the batch (fallback only).
        graph = self._first(prompt, None)
        extra = self._first(extra_pnginfo, None)

        # Per-image metadata lists: map by index, broadcast a single value to all.
        # Each image must carry its *own* original workflow so dragging it back
        # into ComfyUI reproduces that exact image -- a shared workflow would be
        # wrong for every image but one.
        params_list = a1111_params if isinstance(a1111_params, list) else [a1111_params]
        params_list = [p for p in params_list if isinstance(p, str)]
        workflow_json_list = workflow_json if isinstance(workflow_json, list) else [workflow_json]
        workflow_json_list = [w for w in workflow_json_list if isinstance(w, str)]

        post_images = []
        all_individual_images = []

        # Flatten the input list of tensors
        for img_tensor in images:
            if len(img_tensor.shape) == 3:
                all_individual_images.append(img_tensor)
            elif len(img_tensor.shape) == 4:
                for b in range(img_tensor.shape[0]):
                    all_individual_images.append(img_tensor[b])
            else:
                print(f"[Civitai MCP] Warning: Skipped an image tensor with unexpected shape: {img_tensor.shape}")

        num_images = len(all_individual_images)
        if num_images == 0:
            raise ValueError("No valid images passed to Civitai Create Post node.")

        print(f"[Civitai MCP] Processing {num_images} total image(s)...")

        for idx, single_img in enumerate(all_individual_images):
            pil_img = _tensor_to_pil(single_img)
            width, height = pil_img.size

            # Resolve this image's parameters string (broadcast single value,
            # else map by index, else empty when there are more images than entries).
            params_str = self._per_image(params_list, idx).strip()
            if single_embed_metadata and not params_str:
                params_str = civitai_metadata.auto_build_params(graph, width=width, height=height)

            # Resolve this image's workflow the same way; a wired workflow_json
            # (the image's own original graph) wins over the live one.
            workflow_override = _parse_workflow_json(self._per_image(workflow_json_list, idx))

            img_bytes, content_type = civitai_metadata.encode_image(
                pil_img, file_format=single_format, a1111_params=params_str,
                embed_metadata=single_embed_metadata, embed_workflow=single_embed_workflow,
                prompt=graph, extra_pnginfo=extra, jpg_quality=single_quality,
                workflow_override=workflow_override,
            )
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            print(f"[Civitai MCP] Uploading image {idx + 1}/{num_images} ({width}x{height}, {content_type})...")

            upload_res = civitai_api.upload_image(img_base64, content_type=content_type)
            uuid = upload_res.get("uuid")
            if not uuid:
                raise Exception(f"Failed to upload image {idx + 1}: UUID was not returned.")

            post_images.append({
                "uuid": uuid,
                "width": width,
                "height": height,
                "type": "image"
            })
            
        base_title = single_title.strip() if isinstance(single_title, str) and single_title.strip() else None
        final_desc = single_desc.strip() if isinstance(single_desc, str) and single_desc.strip() else None

        # Civitai allows at most 20 images per post; clamp and split into chunks.
        try:
            chunk_size = int(single_max_per_post)
        except (TypeError, ValueError):
            chunk_size = 20
        chunk_size = max(1, min(20, chunk_size))

        chunks = [post_images[i:i + chunk_size] for i in range(0, len(post_images), chunk_size)]
        multiple = len(chunks) > 1
        if multiple:
            print(f"[Civitai MCP] {len(post_images)} images exceed {chunk_size} per post; creating {len(chunks)} posts.")

        post_ids = []
        post_urls = []
        for chunk_idx, chunk in enumerate(chunks):
            # Suffix the title so split posts are distinguishable on the profile.
            if base_title and multiple:
                chunk_title = f"{base_title} ({chunk_idx + 1})"
            else:
                chunk_title = base_title

            print(f"[Civitai MCP] Creating post {chunk_idx + 1}/{len(chunks)} with {len(chunk)} image(s)...")
            post_res = civitai_api.create_post(
                images=chunk,
                title=chunk_title,
                detail=final_desc,
                publish=single_publish,
                model_version_id=single_model_version if single_model_version > 0 else None,
                collection_id=single_collection if single_collection > 0 else None
            )

            post_id = post_res.get("id", 0)
            post_url = post_res.get("url", f"https://civitai.com/posts/{post_id}" if post_id else "")
            post_ids.append(post_id)
            post_urls.append(post_url)
            print(f"[Civitai MCP] Post created successfully! URL: {post_url}")

        # Newline-joined scalar copies for display/save nodes that only read the
        # first element of a list input.
        post_ids_text = "\n".join(str(pid) for pid in post_ids)
        post_urls_text = "\n".join(post_urls)

        return (post_ids, post_urls, post_ids_text, post_urls_text)


DEFAULT_CHECKPOINTS = {
    'SD 1.5':        {'modelId': 4384,    'versionId': 128713},
    'SDXL 1.0':      {'modelId': 101055,  'versionId': 128078},
    'Pony':          {'modelId': 257749,  'versionId': 290640},
    'PonyV7':        {'modelId': 1901521, 'versionId': 2152373},
    'Illustrious':   {'modelId': 795765,  'versionId': 889818},
    'NoobAI':        {'modelId': 833294,  'versionId': 1190596},
    'Flux.1 Dev':    {'modelId': 618692,  'versionId': 691639},
    'Flux.1 Dev (S)':{'modelId': 618692,  'versionId': 691639},
    'Chroma':        {'modelId': 1330309, 'versionId': 2164239},
}

def get_default_checkpoint(base_model_str):
    if not base_model_str:
        return DEFAULT_CHECKPOINTS['SDXL 1.0']
    
    # Fuzzy matching
    for key, val in DEFAULT_CHECKPOINTS.items():
        if key.lower() in base_model_str.lower() or base_model_str.lower() in key.lower():
            return val
            
    bm = base_model_str.lower()
    if 'flux' in bm:
        return DEFAULT_CHECKPOINTS['Flux.1 Dev']
    if 'illustrious' in bm:
        return DEFAULT_CHECKPOINTS['Illustrious']
    if 'pony' in bm:
        return DEFAULT_CHECKPOINTS['Pony']
    if 'sd 1' in bm or 'sd1' in bm:
        return DEFAULT_CHECKPOINTS['SD 1.5']
        
    return DEFAULT_CHECKPOINTS['SDXL 1.0']


class CivitaiGetCurrentChallenge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }
        
    OUTPUT_NODE = True

    RETURN_TYPES = ("INT", "INT", "STRING", "STRING", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("challenge_id", "collection_id", "title", "description", "model_version_id", "model_id", "lora_id", "lora_version_id")
    FUNCTION = "get_challenge"
    CATEGORY = "Civitai-mcp"

    def get_challenge(self):
        import re
        challenge = civitai_api.get_current_challenge()
        
        challenge_id = challenge.get("id", 0)
        collection_id = challenge.get("collectionId", 0)
        title = challenge.get("title", "")
        theme = challenge.get("theme", "")
        
        # Parse description HTML
        desc_html = challenge.get("description", "")
        # Strip HTML tags
        clean_desc = re.sub(r'<[^>]+>', ' ', desc_html)
        clean_desc = re.sub(r'\s+', ' ', clean_desc).strip()
        
        # Combine theme and description for full context/body
        full_desc = f"Theme: {theme}\n\n{clean_desc}" if theme else clean_desc
        
        # Resolve model version ID and model/Lora IDs
        models = challenge.get("models", [])
        challenge_model_version_id = 0
        challenge_model_id = 0
        base_model = "SDXL 1.0"
        
        if models:
            model_entry = models[0]
            challenge_model_version_id = model_entry.get("versionId", 0)
            challenge_model_id = model_entry.get("id", 0)
            base_model = model_entry.get("baseModel", "SDXL 1.0")
        else:
            # Fallback to modelVersionIds list if models list is empty
            version_ids = challenge.get("modelVersionIds", [])
            if version_ids:
                challenge_model_version_id = version_ids[0]
                
        model_version_id = 0
        model_id = 0
        lora_id = 0
        lora_version_id = 0
        
        if challenge_model_version_id > 0:
            m_type = civitai_api.get_model_version_type(challenge_model_version_id)
            if m_type in ["LORA", "LOCON", "DORA"]:
                lora_id = challenge_model_id
                lora_version_id = challenge_model_version_id
                
                # Retrieve default fallback checkpoint for the base model
                default_chk = get_default_checkpoint(base_model)
                model_id = default_chk['modelId']
                model_version_id = default_chk['versionId']
            else:
                model_id = challenge_model_id
                model_version_id = challenge_model_version_id
                
        print(f"[Civitai MCP] Challenge retrieved: '{title}' (ID: {challenge_id}, Collection: {collection_id})")
        
        return (challenge_id, collection_id, title, full_desc, model_version_id, model_id, lora_id, lora_version_id)


class CivitaiGetModelMetadata:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_version_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("INT", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model_id", "model_name", "version_name", "base_model", "trigger_words", "air_urn", "download_url")
    FUNCTION = "get_metadata"
    CATEGORY = "Civitai-mcp"

    def get_metadata(self, model_version_id):
        if model_version_id <= 0:
            return (0, "", "", "", "", "", "")
            
        data = civitai_api.get_model_version_metadata(model_version_id)
        model_id = data.get("modelId", 0)
        version_name = data.get("name", "")
        base_model = data.get("baseModel", "")
        air_urn = data.get("air", "")
        download_url = data.get("downloadUrl", "")
        
        # Extract model name
        model_data = data.get("model", {})
        model_name = model_data.get("name", "")
        
        # Trained words (trigger words) are returned as a list of strings
        trained_words = data.get("trainedWords", [])
        trigger_words = ", ".join(trained_words) if trained_words else ""
        
        print(f"[Civitai MCP] Retrieved model metadata for '{model_name}' ({version_name})")
        return (model_id, model_name, version_name, base_model, trigger_words, air_urn, download_url)


class CivitaiGetImageMetadata:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_rating": (["PG", "PG-13", "R", "X", "XXX"], {"default": "XXX"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt", "seed", "steps", "cfg_scale", "sampler_name", "width", "height", "tags")
    FUNCTION = "get_metadata"
    CATEGORY = "Civitai-mcp"

    def get_metadata(self, image_id, max_rating="XXX"):
        if image_id <= 0:
            return ("", "", 0, 0, 0.0, "", 0, 0, "")

        data = civitai_api.get_image_metadata(image_id, nsfw_level=max_rating)

        # The API returns generation parameters directly under "meta" (flatMeta=true)
        meta = data.get("meta") or {}

        # Top-level width/height can be null; fall back to the values in meta
        width = data.get("width") or meta.get("width", 0)
        height = data.get("height") or meta.get("height", 0)

        prompt = meta.get("prompt", "")
        negative_prompt = meta.get("negativePrompt", "")
        seed = meta.get("seed", 0)
        steps = meta.get("steps", 0)
        cfg_scale = float(meta.get("cfgScale", 0.0))
        sampler_name = meta.get("sampler", "")

        # Tags come from withTags=true as a list of {id, name}; expose names as a comma-separated string
        tags = ", ".join(t.get("name", "") for t in (data.get("tags") or []) if t.get("name"))

        print(f"[Civitai MCP] Retrieved image metadata for ID {image_id} (max rating: {max_rating})")
        return (prompt, negative_prompt, seed, steps, cfg_scale, sampler_name, width, height, tags)


class CivitaiAccountStatus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("username", "user_id", "is_moderator", "is_onboarded", "muted")
    FUNCTION = "get_status"
    CATEGORY = "Civitai-mcp"

    def get_status(self):
        res = civitai_api.call_mcp_tool("whoami", {})
        structured = res.get("structuredContent", {})
        
        username = structured.get("username", "")
        user_id = structured.get("id", 0)
        is_moderator = structured.get("isModerator", False)
        is_onboarded = structured.get("isOnboarded", False)
        muted = structured.get("muted", False)
        
        print(f"[Civitai MCP] Account Status: {username} (ID: {user_id})")
        return (username, user_id, is_moderator, is_onboarded, muted)


class CivitaiGetImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "max_rating": (["PG", "PG-13", "R", "X", "XXX"], {"default": "XXX"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "get_image"
    CATEGORY = "Civitai-mcp"

    def get_image(self, image_id, max_rating="XXX"):
        if image_id <= 0:
            # Return an empty/black 64x64 placeholder tensor if no valid ID
            empty_img = torch.zeros([1, 64, 64, 3], dtype=torch.float32)
            return (empty_img,)
            
        img_bytes = civitai_api.download_image_by_id(image_id, nsfw_level=max_rating)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        import numpy as np
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)
        
        print(f"[Civitai MCP] Successfully retrieved and loaded image {image_id} (max rating: {max_rating})")
        return (img_tensor,)


