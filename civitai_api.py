import json
import urllib.request
import urllib.parse
import urllib.error
import os

def call_mcp_tool(tool_name, arguments, api_key=None):
    """
    Calls a Civitai MCP server tool using JSON-RPC over HTTP.
    """
    url = "https://mcp.civitai.com/mcp"
    
    # Resolve API Key
    if not api_key:
        # Check for local text file first
        current_dir = os.path.dirname(os.path.abspath(__file__))
        key_file_path = os.path.join(current_dir, "civitai_key.txt")
        if os.path.exists(key_file_path):
            try:
                with open(key_file_path, "r", encoding="utf-8") as f:
                    api_key = f.read().strip()
            except Exception as e:
                print(f"[Civitai MCP] Warning: Failed to read {key_file_path}: {e}")
                
        # Fallback to environment variable if still empty
        if not api_key:
            api_key = os.environ.get("CIVITAI_API_KEY", "")
    
    api_key = api_key.strip()
    if not api_key:
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

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "User-Agent": "ComfyUI-Civitai-MCP/1.0",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        },
        "id": 1
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers=headers,
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as response:
            resp_str = response.read().decode('utf-8')
            try:
                data = json.loads(resp_str)
            except json.JSONDecodeError:
                # Fallback to parsing event-stream response
                lines = resp_str.splitlines()
                json_data = None
                for line in lines:
                    if line.startswith("data:"):
                        data_content = line[5:].strip()
                        try:
                            json_data = json.loads(data_content)
                            break
                        except:
                            pass
                if json_data:
                    data = json_data
                else:
                    raise ValueError(f"Failed to parse server response: {resp_str}")
            
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
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        try:
            error_json = json.loads(error_body)
            # If the server returned a JSON-RPC error structure
            if "error" in error_json:
                msg = error_json["error"].get("message", error_body)
            else:
                msg = error_json.get("error", error_body)
        except:
            msg = error_body
        raise Exception(f"Civitai API Error (HTTP {e.code}): {msg}")
    except Exception as e:
        raise Exception(f"Connection Error: {e}")


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
        "contentType": content_type
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
        "publish": publish
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
    # Resolve API Key
    if not api_key:
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
            
    api_key = api_key.strip()
    if not api_key:
        raise ValueError(
            "Civitai API Key is required to call tRPC procedure.\n"
            "Please create a file named 'civitai_key.txt' containing your key inside the custom node directory, "
            "or set the CIVITAI_API_KEY environment variable."
        )

    json_input = {"json": input_data}
    encoded_input = urllib.parse.quote(json.dumps(json_input))
    url = f"https://civitai.com/api/trpc/{procedure}?input={encoded_input}"
    
    headers = {
        "User-Agent": "ComfyUI-Civitai-MCP/1.0",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req) as response:
            resp_str = response.read().decode('utf-8')
            data = json.loads(resp_str)
            return data.get("result", {}).get("data", {}).get("json", {})
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        try:
            error_json = json.loads(error_body)
            msg = error_json.get("error", {}).get("message", error_body)
        except:
            msg = error_body
        raise Exception(f"TRPC Call {procedure} Failed (HTTP {e.code}): {msg}")
    except Exception as e:
        raise Exception(f"TRPC Call {procedure} Connection Error: {e}")


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


def get_model_version_type(version_id, api_key=None):
    """
    Fetches the model type for a given model version ID using the public REST API.
    """
    url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    headers = {
        "User-Agent": "ComfyUI-Civitai-MCP/1.0",
        "Accept": "application/json"
    }
    # Resolve API Key if not provided explicitly
    if not api_key:
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
            
    api_key = api_key.strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            model = data.get("model", {})
            return model.get("type", "").upper()
    except Exception as e:
        print(f"[Civitai MCP] Warning: Failed to query model version {version_id} type: {e}")
        return ""


def get_model_version_metadata(version_id, api_key=None):
    """
    Fetches detailed metadata for a specific model version using the public REST API.
    """
    url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    headers = {
        "User-Agent": "ComfyUI-Civitai-MCP/1.0",
        "Accept": "application/json"
    }
    # Resolve API Key if not provided explicitly
    if not api_key:
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
            
    api_key = api_key.strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode('utf-8'))
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
        "XXX": ("X", 31)
    }
    nsfw_val, level_val = rating_map.get(nsfw_level, ("X", 31))
    url = f"https://civitai.com/api/v1/images?imageId={image_id}&withMeta=true&flatMeta=true&withTags=true&nsfw={nsfw_val}&browsingLevel={level_val}"
    headers = {
        "User-Agent": "ComfyUI-Civitai-MCP/1.0",
        "Accept": "application/json"
    }
    # Resolve API Key if not provided explicitly
    if not api_key:
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
            
    api_key = api_key.strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
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
    except Exception as e:
        raise e


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
        
    headers = {
        "User-Agent": "ComfyUI-Civitai-MCP/1.0"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            return response.read()
    except Exception as e:
        raise Exception(f"Failed to download image from {url}: {e}")




