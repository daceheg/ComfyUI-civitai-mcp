import io
import base64
import torch
from PIL import Image
from . import civitai_api

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
            }
        }

    OUTPUT_NODE = True

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("post_id", "post_url")
    FUNCTION = "post_image"
    CATEGORY = "Civitai"

    def post_image(self, image, publish, title="", description="", model_version_id=0, collection_id=0):
        # ComfyUI images are tensors with shape [B, H, W, C]
        # We only take the first image if it's a batch
        if len(image.shape) == 4 and image.shape[0] > 1:
            print(f"[Civitai MCP] Warning: 'Civitai Post Image' received a batch of {image.shape[0]} images. Only the first image will be posted. Use 'Civitai Create Post' for multiple images.")
        
        single_img = image[0]
        
        # Convert to numpy uint8
        img_np = (single_img.cpu().numpy() * 255).clip(0, 255).astype('uint8')
        pil_img = Image.fromarray(img_np)
        
        # Save PIL image to base64
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        width, height = pil_img.size
        print(f"[Civitai MCP] Uploading single image ({width}x{height})...")
        
        # Upload image to Civitai via MCP
        upload_res = civitai_api.upload_image(img_base64, content_type="image/png")
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
                "model_version_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "collection_id": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    INPUT_IS_LIST = True

    OUTPUT_NODE = True

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("post_id", "post_url")
    FUNCTION = "create_post"
    CATEGORY = "Civitai"

    def create_post(self, images, publish, title, description, model_version_id, collection_id):
        # Extract scalar options from lists
        single_publish = publish[0] if isinstance(publish, list) and len(publish) > 0 else True
        single_title = title[0] if isinstance(title, list) and len(title) > 0 else ""
        single_desc = description[0] if isinstance(description, list) and len(description) > 0 else ""
        single_model_version = model_version_id[0] if isinstance(model_version_id, list) and len(model_version_id) > 0 else 0
        single_collection = collection_id[0] if isinstance(collection_id, list) and len(collection_id) > 0 else 0

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
            
        print(f"[Civitai MCP] Processing {num_images} total images for a single post...")
        
        for idx, single_img in enumerate(all_individual_images):
            img_np = (single_img.cpu().numpy() * 255).clip(0, 255).astype('uint8')
            pil_img = Image.fromarray(img_np)
            
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            width, height = pil_img.size
            print(f"[Civitai MCP] Uploading image {idx + 1}/{num_images} ({width}x{height})...")
            
            upload_res = civitai_api.upload_image(img_base64, content_type="image/png")
            uuid = upload_res.get("uuid")
            if not uuid:
                raise Exception(f"Failed to upload image {idx + 1}: UUID was not returned.")
                
            post_images.append({
                "uuid": uuid,
                "width": width,
                "height": height,
                "type": "image"
            })
            
        final_title = single_title.strip() if isinstance(single_title, str) and single_title.strip() else None
        final_desc = single_desc.strip() if isinstance(single_desc, str) and single_desc.strip() else None
        
        print(f"[Civitai MCP] Creating multi-image post with {len(post_images)} images...")
        post_res = civitai_api.create_post(
            images=post_images,
            title=final_title,
            detail=final_desc,
            publish=single_publish,
            model_version_id=single_model_version if single_model_version > 0 else None,
            collection_id=single_collection if single_collection > 0 else None
        )
        
        post_id = post_res.get("id", 0)
        post_url = post_res.get("url", f"https://civitai.com/posts/{post_id}" if post_id else "")
        
        print(f"[Civitai MCP] Multi-image post created successfully! URL: {post_url}")
        
        return (post_id, post_url)


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
    CATEGORY = "Civitai"

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
    CATEGORY = "Civitai"

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

    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "FLOAT", "STRING", "INT", "INT")
    RETURN_NAMES = ("prompt", "negative_prompt", "seed", "steps", "cfg_scale", "sampler_name", "width", "height")
    FUNCTION = "get_metadata"
    CATEGORY = "Civitai"

    def get_metadata(self, image_id, max_rating="XXX"):
        if image_id <= 0:
            return ("", "", 0, 0, 0.0, "", 0, 0)
            
        data = civitai_api.get_image_metadata(image_id, nsfw_level=max_rating)
        width = data.get("width", 0)
        height = data.get("height", 0)
        
        meta_root = data.get("meta") or {}
        # The REST API nests the generation parameters under a secondary "meta" key
        meta = meta_root.get("meta") if isinstance(meta_root.get("meta"), dict) else meta_root
        
        prompt = meta.get("prompt", "")
        negative_prompt = meta.get("negativePrompt", "")
        seed = meta.get("seed", 0)
        steps = meta.get("steps", 0)
        cfg_scale = float(meta.get("cfgScale", 0.0))
        sampler_name = meta.get("sampler", "")
        
        print(f"[Civitai MCP] Retrieved image metadata for ID {image_id} (max rating: {max_rating})")
        return (prompt, negative_prompt, seed, steps, cfg_scale, sampler_name, width, height)


class CivitaiAccountStatus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {}
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("username", "user_id", "is_moderator", "is_onboarded", "muted")
    FUNCTION = "get_status"
    CATEGORY = "Civitai"

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
    CATEGORY = "Civitai"

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


