import os
import json

import requests

# Shared timeout (seconds) for Civitai API calls.
_TIMEOUT = 60
_USER_AGENT = "ComfyUI-Civitai-MCP/1.0"


def _resolve_api_key(api_key=None):
    """Resolve the Civitai API key.

    Order: explicit argument, then a local ``civitai_key.txt`` next to this
    module, then the ``CIVITAI_API_KEY`` environment variable. Returns the
    stripped key, or "" when none is configured.
    """
    if api_key:
        return api_key.strip()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    key_file_path = os.path.join(current_dir, "civitai_key.txt")
    if os.path.exists(key_file_path):
        try:
            with open(key_file_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
        except Exception as e:
            print(f"[Civitai MCP] Warning: Failed to read {key_file_path}: {e}")

    if not api_key:
        api_key = os.environ.get("CIVITAI_API_KEY", "")

    return (api_key or "").strip()


def _require_api_key(api_key=None):
    """Resolve the API key, raising a helpful error if it isn't configured."""
    key = _resolve_api_key(api_key)
    if not key:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        key_file_path = os.path.join(current_dir, "civitai_key.txt")
        raise ValueError(
            f"\n\n[Civitai MCP Error] Civitai API Key is missing!\n"
            f"-----------------------------------------------------------------\n"
            f"To fix this, please create a text file named 'civitai_key.txt' at:\n"
            f"  {key_file_path}\n\n"
            f"Open this file, paste your Civitai API key (get one from https://civitai.com/user/account),\n"
            f"save it, and restart ComfyUI.\n\n"
            f"Alternatively, you can define the 'CIVITAI_API_KEY' environment variable.\n"
            f"-----------------------------------------------------------------\n"
        )
    return key


def call_mcp_tool(tool_name, arguments, api_key=None):
    """
    Calls a Civitai MCP server tool using JSON-RPC over HTTP.
    """
    url = "https://mcp.civitai.com/mcp"
    api_key = _require_api_key(api_key)

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": _USER_AGENT,
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
        "id": 1,
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=_TIMEOUT)
    except requests.RequestException as e:
        raise Exception(f"Connection Error: {e}")

    resp_str = response.text
    if not response.ok:
        # Surface a JSON-RPC / API error message when present.
        msg = resp_str
        try:
            error_json = json.loads(resp_str)
            if "error" in error_json:
                msg = error_json["error"].get("message", resp_str)
            else:
                msg = error_json.get("error", resp_str)
        except json.JSONDecodeError:
            pass
        raise Exception(f"Civitai API Error (HTTP {response.status_code}): {msg}")

    try:
        data = json.loads(resp_str)
    except json.JSONDecodeError:
        # Fallback to parsing an event-stream response.
        json_data = None
        for line in resp_str.splitlines():
            if line.startswith("data:"):
                try:
                    json_data = json.loads(line[5:].strip())
                    break
                except json.JSONDecodeError:
                    pass
        if json_data is None:
            raise ValueError(f"Failed to parse server response: {resp_str}")
        data = json_data

    # Check for JSON-RPC error
    if "error" in data:
        raise Exception(data["error"].get("message", "Unknown server error"))

    result = data.get("result", {})

    # Check for MCP execution errors
    if result.get("isError", False):
        content_items = result.get("content", [])
        err_msg = ""
        for item in content_items:
            if item.get("type") == "text":
                err_msg += item.get("text", "") + "\n"
        raise Exception(f"MCP Tool Execution Error: {err_msg.strip() or 'Unknown error'}")

    return result


def upload_image(base64_data, content_type="image/png", api_key=None):
    """
    Uploads base64-encoded image bytes to Civitai.
    Returns: dict with 'uuid', 'width', 'height'
    """
    # Remove data:image/...;base64, prefix if present
    if "," in base64_data:
        base64_data = base64_data.split(",", 1)[1]

    arguments = {
        "data": base64_data,
        "contentType": content_type,
    }

    result = call_mcp_tool("upload_image", arguments, api_key=api_key)
    structured = result.get("structuredContent")
    if structured and isinstance(structured, dict):
        return structured

    # If structuredContent is not returned, parse the text content
    content_list = result.get("content", [])
    for item in content_list:
        if item.get("type") == "text":
            text = item.get("text", "")
            # Example text fallback: "Uploaded image successfully. UUID: <uuid>"
            if "UUID:" in text:
                parts = text.split("UUID:")
                uuid = parts[1].strip().split()[0]
                return {"uuid": uuid}

    raise Exception("Failed to retrieve image UUID from upload response.")


def create_post(images, title=None, detail=None, publish=True, api_key=None, model_version_id=None, collection_id=None, tags=None):
    """
    Creates a Civitai post containing multiple images.
    images: List of dicts, each with keys: 'uuid', 'width', 'height', 'type'
    """
    arguments = {
        "images": images,
        "publish": publish,
    }
    if title:
        arguments["title"] = title
    if detail:
        arguments["detail"] = detail
    if model_version_id and model_version_id > 0:
        arguments["modelVersionId"] = model_version_id
    if collection_id and collection_id > 0:
        arguments["collectionId"] = collection_id
    if tags:
        arguments["tags"] = tags

    result = call_mcp_tool("create_post", arguments, api_key=api_key)
    structured = result.get("structuredContent")
    if structured and isinstance(structured, dict):
        return structured

    # Fallback to parsing text
    content_list = result.get("content", [])
    for item in content_list:
        if item.get("type") == "text":
            text = item.get("text", "")
            # Find URLs or IDs in text if needed
            return {"text": text}

    return result


def call_trpc(procedure, input_data, api_key=None):
    """
    Queries a Civitai internal tRPC procedure.
    """
    api_key = _resolve_api_key(api_key)
    if not api_key:
        raise ValueError(
            "Civitai API Key is required to call tRPC procedure.\n"
            "Please create a file named 'civitai_key.txt' containing your key inside the custom node directory, "
            "or set the CIVITAI_API_KEY environment variable."
        )

    url = f"https://civitai.com/api/trpc/{procedure}"
    params = {"input": json.dumps({"json": input_data})}
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)
    except requests.RequestException as e:
        raise Exception(f"TRPC Call {procedure} Connection Error: {e}")

    if not response.ok:
        msg = response.text
        try:
            msg = response.json().get("error", {}).get("message", response.text)
        except ValueError:
            pass
        raise Exception(f"TRPC Call {procedure} Failed (HTTP {response.status_code}): {msg}")

    data = response.json()
    return data.get("result", {}).get("data", {}).get("json", {})


def get_current_challenge(api_key=None):
    """
    Queries the active daily challenge and returns its details.
    """
    # 1. Get active challenge ID
    infinite_res = call_trpc("challenge.getInfinite", {"status": ["Active"], "limit": 1}, api_key=api_key)
    items = infinite_res.get("items", [])
    if not items:
        raise Exception("No active challenge found on Civitai.")

    challenge_id = items[0].get("id")

    # 2. Get full details of this challenge
    challenge_details = call_trpc("challenge.getById", {"id": challenge_id}, api_key=api_key)
    return challenge_details


def _rest_headers(api_key=None, accept="application/json"):
    """Build REST headers, adding a Bearer token when a key is available."""
    headers = {"User-Agent": _USER_AGENT}
    if accept:
        headers["Accept"] = accept
    api_key = _resolve_api_key(api_key)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def get_model_version_type(version_id, api_key=None):
    """
    Fetches the model type for a given model version ID using the public REST API.
    """
    url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    try:
        response = requests.get(url, headers=_rest_headers(api_key), timeout=_TIMEOUT)
        response.raise_for_status()
        model = response.json().get("model", {})
        return model.get("type", "").upper()
    except Exception as e:
        print(f"[Civitai MCP] Warning: Failed to query model version {version_id} type: {e}")
        return ""


def get_model_version_metadata(version_id, api_key=None):
    """
    Fetches detailed metadata for a specific model version using the public REST API.
    """
    url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    try:
        response = requests.get(url, headers=_rest_headers(api_key), timeout=_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"Failed to query model version {version_id}: {e}")


def get_image_metadata(image_id, nsfw_level="XXX", api_key=None):
    """
    Fetches detailed metadata for a specific image using the public REST API.
    """
    rating_map = {
        "PG": ("None", 1),
        "PG-13": ("Soft", 3),
        "R": ("Mature", 7),
        "X": ("X", 15),
        "XXX": ("X", 31),
    }
    nsfw_val, level_val = rating_map.get(nsfw_level, ("X", 31))
    url = "https://civitai.com/api/v1/images"
    params = {
        "imageId": image_id,
        "withMeta": "true",
        "flatMeta": "true",
        "withTags": "true",
        "nsfw": nsfw_val,
        "browsingLevel": level_val,
    }

    response = requests.get(url, params=params, headers=_rest_headers(api_key), timeout=_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    items = data.get("items", [])
    if not items:
        if nsfw_level != "XXX":
            try:
                fallback_data = get_image_metadata(image_id, nsfw_level="XXX", api_key=api_key)
                if fallback_data:
                    raise Exception(
                        f"Image ID {image_id} exists but exceeds your selected max_rating ('{nsfw_level}'). "
                        f"Try increasing the max_rating to a higher level (e.g., 'X' or 'XXX')."
                    )
            except Exception as fallback_err:
                if "exceeds your selected max_rating" in str(fallback_err):
                    raise fallback_err
        raise Exception(f"No image found with ID {image_id}")
    return items[0]


def get_image_url(image_id, nsfw_level="XXX", api_key=None):
    """
    Fetches the URL of a specific image using the public REST API.
    """
    data = get_image_metadata(image_id, nsfw_level=nsfw_level, api_key=api_key)
    return data.get("url")


def download_image_by_id(image_id, nsfw_level="XXX", api_key=None):
    """
    Downloads the image bytes for a specific image ID from Civitai.
    """
    url = get_image_url(image_id, nsfw_level=nsfw_level, api_key=api_key)
    if not url:
        raise Exception(f"No URL found for image ID {image_id}")

    try:
        response = requests.get(url, headers={"User-Agent": _USER_AGENT}, timeout=_TIMEOUT)
        response.raise_for_status()
        return response.content
    except Exception as e:
        raise Exception(f"Failed to download image from {url}: {e}")
