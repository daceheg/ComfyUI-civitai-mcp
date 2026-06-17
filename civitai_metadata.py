"""
Generation-metadata auto-detection and image encoding for Civitai posts.

This module reconstructs an A1111-style ``parameters`` string from the ComfyUI
prompt graph (the hidden ``PROMPT`` input) so that images uploaded to Civitai
carry full generation metadata -- prompt, sampler settings, and a fully
resolved ``Civitai resources`` array (AIR URNs + weights) -- without the user
having to wire anything.

The resource resolution mirrors ComfyUI-Image-Saver: model/LoRA filenames are
resolved via ``folder_paths``, hashed (SHA256), and looked up against Civitai's
``model-versions/by-hash`` endpoint, which returns a ready-made ``air`` URN.
Hashes are cached both in-memory and as ``.civitai.info`` sidecar files next to
the model (Image-Saver-compatible) to avoid re-hashing multi-GB checkpoints.

If the caller supplies an explicit ``a1111_params`` string (e.g. wired from
Image Saver's output) it is used verbatim and this auto-detection is skipped.
"""

import io
import os
import re
import json
import hashlib
import urllib.request
import urllib.error

try:
    import folder_paths  # provided by ComfyUI at runtime
except Exception:  # pragma: no cover - allows host-side unit testing
    folder_paths = None

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PIL.ExifTags import IFD

# EXIF UserComment tag id; A1111/Civitai store the parameters string here for JPEG.
_EXIF_USER_COMMENT = 0x9286

# ComfyUI sampler id -> Civitai display name. Mirrors ComfyUI-Image-Saver's map
# (https://github.com/civitai/civitai/blob/main/src/server/common/constants.ts).
CIVITAI_SAMPLER_MAP = {
    "euler_ancestral": "Euler a",
    "euler": "Euler",
    "lms": "LMS",
    "heun": "Heun",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_3m_sde": "DPM++ 3M SDE",
    "dpm_fast": "DPM fast",
    "dpm_adaptive": "DPM adaptive",
    "ddim": "DDIM",
    "plms": "PLMS",
    "uni_pc_bh2": "UniPC",
    "uni_pc": "UniPC",
    "lcm": "LCM",
}

# <lora:NAME:weight> with optional extra block-weight args after the weight.
_LORA_RE = re.compile(r"<lora:([^>:]+)(?::([^>]+))?>", re.IGNORECASE)

# In-memory hash cache for the lifetime of the process: abspath -> sha256 hex.
_HASH_CACHE = {}
# In-memory AIR cache: sha256 hex -> resource dict (or None for a known miss).
_AIR_CACHE = {}


def civitai_sampler_name(sampler, scheduler):
    """Map a ComfyUI sampler/scheduler pair to the Civitai display name."""
    name = CIVITAI_SAMPLER_MAP.get(sampler, sampler)
    if sampler in CIVITAI_SAMPLER_MAP:
        if scheduler == "karras":
            name += " Karras"
        elif scheduler == "exponential":
            name += " Exponential"
    elif scheduler and scheduler != "normal":
        name = f"{sampler}_{scheduler}"
    return name


def _sha256_file(path):
    """SHA256 a file, caching by absolute path. Returns hex digest or None."""
    try:
        abspath = os.path.abspath(path)
    except Exception:
        return None
    if abspath in _HASH_CACHE:
        return _HASH_CACHE[abspath]
    try:
        h = hashlib.sha256()
        with open(abspath, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        digest = h.hexdigest()
        _HASH_CACHE[abspath] = digest
        return digest
    except Exception as e:
        print(f"[Civitai MCP] Warning: failed to hash {path}: {e}")
        return None


def _sidecar_path(model_path):
    return os.path.splitext(model_path)[0] + ".civitai.info"


def _read_sidecar(model_path):
    """Read a cached .civitai.info blob next to the model, if present."""
    sidecar = _sidecar_path(model_path)
    try:
        if os.path.exists(sidecar):
            with open(sidecar, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[Civitai MCP] Warning: failed to read sidecar {sidecar}: {e}")
    return None


def _write_sidecar(model_path, blob):
    """Cache a by-hash response as .civitai.info next to the model file."""
    sidecar = _sidecar_path(model_path)
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            json.dump(blob, f, indent=2)
    except Exception as e:
        # Read-only model dir, permissions, etc. -- non-fatal, in-memory cache still helps.
        print(f"[Civitai MCP] Note: could not write sidecar {sidecar}: {e}")


def _fetch_by_hash(sha256, api_key=None):
    """Look up a model version on Civitai by SHA256. Returns the JSON blob or None."""
    url = f"https://civitai.com/api/v1/model-versions/by-hash/{sha256.upper()}"
    headers = {
        "User-Agent": "ComfyUI-Civitai-MCP/1.0",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # model not on Civitai -- expected, not an error
        print(f"[Civitai MCP] Warning: by-hash lookup failed (HTTP {e.code}) for {sha256[:10]}")
    except Exception as e:
        print(f"[Civitai MCP] Warning: by-hash lookup error for {sha256[:10]}: {e}")
    return None


def _resource_from_blob(blob, weight=None):
    """Build a Civitai-resources entry from a by-hash/sidecar blob."""
    if not isinstance(blob, dict):
        return None
    resource = {}
    model = blob.get("model") or {}
    if model.get("name"):
        resource["modelName"] = model["name"]
    if blob.get("name"):
        resource["versionName"] = blob["name"]
    if weight is not None:
        resource["weight"] = weight
    # Prefer the ready-made AIR URN; fall back to the version id.
    if blob.get("air"):
        resource["air"] = blob["air"]
    elif blob.get("id"):
        resource["modelVersionId"] = blob["id"]
    else:
        return None
    return resource


def resolve_resource(filename, folder_type, weight=None, api_key=None):
    """
    Resolve a model/LoRA filename to a Civitai-resources entry.

    filename: name as it appears in the graph (may include a subfolder).
    folder_type: a folder_paths key, e.g. "checkpoints" or "loras".
    Returns a resource dict (with air/modelVersionId) or None if unresolved.
    """
    if folder_paths is None:
        return None
    try:
        path = folder_paths.get_full_path(folder_type, filename)
    except Exception:
        path = None
    if not path or not os.path.exists(path):
        return None

    # Sidecar cache first -- lets us skip hashing entirely on a warm cache.
    blob = _read_sidecar(path)
    if blob is not None:
        return _resource_from_blob(blob, weight)

    digest = _sha256_file(path)
    if not digest:
        return None

    if digest in _AIR_CACHE:
        cached = _AIR_CACHE[digest]
        return _resource_from_blob(cached, weight) if cached else None

    blob = _fetch_by_hash(digest, api_key=api_key)
    _AIR_CACHE[digest] = blob
    if blob is None:
        return None
    _write_sidecar(path, blob)
    return _resource_from_blob(blob, weight)


def _iter_nodes(prompt_graph):
    """Yield (node_id, class_type, inputs) for each node in an API-format prompt."""
    if not isinstance(prompt_graph, dict):
        return
    for node_id, node in prompt_graph.items():
        if isinstance(node, dict):
            yield node_id, node.get("class_type", ""), node.get("inputs", {}) or {}


# Output field names that commonly hold the resolved value on primitive/literal
# / passthrough nodes, so we know which widget to read when chasing a link.
_VALUE_WIDGET_KEYS = (
    "value", "Value", "number", "int", "float", "seed", "string", "text",
    "STRING", "INT", "FLOAT",
)


def _resolve_concat(inputs, prompt_graph, depth):
    """Resolve a CR-Prompt-List / StringConcatenate-style text builder.

    Concatenates prepend/multiline/append (or string_a + delimiter + string_b)
    after resolving each part. Returns the joined string, or None if no part
    resolves. Common in prompt pipelines where text is assembled upstream.
    """
    def _part(key):
        if key not in inputs:
            return None
        val = inputs[key]
        resolved = _resolve_link(val, prompt_graph, depth + 1) if isinstance(val, list) else val
        return resolved if isinstance(resolved, str) else None

    # CR Prompt List style
    if any(k in inputs for k in ("prepend_text", "multiline_text", "append_text")):
        pieces = [_part("prepend_text"), _part("multiline_text"), _part("append_text")]
        pieces = [p for p in pieces if p]
        return "".join(pieces) if pieces else None

    # StringConcatenate style: string_a + delimiter + string_b
    if "string_a" in inputs or "string_b" in inputs:
        a = _part("string_a") or ""
        b = _part("string_b") or ""
        delim = _part("delimiter") or ""
        joined = f"{a}{delim}{b}" if (a or b) else ""
        return joined or None

    return None


def _resolve_link(value, prompt_graph, _depth=0):
    """
    Resolve an input value that may be a ``[node_id, output_index]`` link.

    ComfyUI's API-format graph stores wired inputs as link references rather than
    literals. Walk back through primitive/literal/passthrough and common
    text-builder nodes until a concrete literal is found. Returns the literal
    (number/str) or None if it can't be resolved from the static graph.
    """
    if not isinstance(value, list) or len(value) != 2 or _depth > 16:
        return value if not isinstance(value, list) else None
    node_id = value[0]
    node = prompt_graph.get(node_id) if isinstance(prompt_graph, dict) else None
    if not isinstance(node, dict):
        return None
    ctype = node.get("class_type", "").lower()
    inputs = node.get("inputs", {}) or {}

    # ShowText / preview-text nodes cache their resolved (runtime) text back into
    # the executed graph as text_0/text. This is often the only place a prompt
    # produced by an upstream LLM or other runtime node is statically readable.
    if "showtext" in ctype or "displaytext" in ctype or "previewtext" in ctype:
        for key in ("text_0", "text", "string"):
            cached = inputs.get(key)
            if isinstance(cached, str) and cached.strip():
                return cached
            if isinstance(cached, list):
                resolved = _resolve_link(cached, prompt_graph, _depth + 1)
                if isinstance(resolved, str) and resolved.strip():
                    return resolved

    # Text-builder nodes (CR Prompt List, StringConcatenate, ...) assemble their
    # output from several inputs rather than exposing a single widget value.
    concat = _resolve_concat(inputs, prompt_graph, _depth)
    if concat is not None:
        return concat

    # Primitive / literal nodes expose their value under a known widget key.
    for key in _VALUE_WIDGET_KEYS:
        if key in inputs:
            inner = inputs[key]
            if isinstance(inner, list):
                return _resolve_link(inner, prompt_graph, _depth + 1)
            if isinstance(inner, (str, int, float)) and not isinstance(inner, bool):
                return inner
    # Single-input passthrough (e.g. reroute): follow the lone link.
    literal_inputs = [v for v in inputs.values()]
    if len(literal_inputs) == 1 and isinstance(literal_inputs[0], list):
        return _resolve_link(literal_inputs[0], prompt_graph, _depth + 1)
    return None


def _get(inputs, key, prompt_graph):
    """Read an input, resolving a link reference to its literal value if needed."""
    if key not in inputs:
        return None
    value = inputs[key]
    if isinstance(value, list):
        return _resolve_link(value, prompt_graph)
    return value


def _get_loras_value(inputs):
    """Extract the LoRA Manager loras list from a node's inputs.

    LoRA Manager stores active loras as ``{"loras": {"__value__": [...]}}`` (new)
    or ``{"loras": [...]}`` (old); each entry is ``{"name", "strength", ...}``.
    Returns a list of entry dicts (empty if absent/unrecognized).
    """
    loras = inputs.get("loras")
    if isinstance(loras, dict) and "__value__" in loras:
        loras = loras["__value__"]
    if isinstance(loras, list):
        return [e for e in loras if isinstance(e, dict)]
    return []


def extract_resources(prompt_graph, positive_text="", api_key=None):
    """
    Build the Civitai-resources list from the prompt graph.

    Sources, in order:
      1. checkpoint / diffusion-model loader nodes -> ckpt/unet filename
      2. LoRA loader nodes -> lora_name, or LoRA Manager's loras.__value__ list
      3. inline <lora:name:weight> tags in the positive prompt text
    """
    resources = []
    seen = set()  # avoid duplicate AIRs

    def _add(resource):
        if not resource:
            return
        key = resource.get("air") or resource.get("modelVersionId")
        if key and key not in seen:
            seen.add(key)
            resources.append(resource)

    for _id, ctype, inputs in _iter_nodes(prompt_graph):
        ct = ctype.lower()
        # Checkpoint / diffusion-model loaders
        if "checkpoint" in ct or "unet" in ct or "diffusionmodel" in ct:
            name = _get(inputs, "ckpt_name", prompt_graph) or _get(inputs, "unet_name", prompt_graph)
            if isinstance(name, str):
                folder = "checkpoints" if "checkpoint" in ct else "diffusion_models"
                _add(resolve_resource(name, folder, api_key=api_key))
        # LoRA loader nodes
        elif "lora" in ct:
            # Standard loaders expose a single lora_name + strength_model.
            name = _get(inputs, "lora_name", prompt_graph)
            if isinstance(name, str):
                weight = _get(inputs, "strength_model", prompt_graph)
                if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                    weight = None
                _add(resolve_resource(name, "loras", weight=weight, api_key=api_key))
            # LoRA Manager nodes carry a list under loras.__value__:
            # [{"name", "strength", "clipStrength"?}, ...]
            for entry in _get_loras_value(inputs):
                lname = entry.get("name")
                if not isinstance(lname, str):
                    continue
                w = entry.get("strength")
                w = w if isinstance(w, (int, float)) and not isinstance(w, bool) else None
                _add(resolve_resource(lname, "loras", weight=w, api_key=api_key))

    # Inline <lora:...> tags can appear in the prompt text or, for LoRA Manager
    # nodes, in the autocomplete meta-text widget (insertedText/textSnapshot).
    lora_text_blobs = [positive_text or ""]
    for _id, ctype, inputs in _iter_nodes(prompt_graph):
        meta = inputs.get("__lm_autocomplete_meta_text")
        if isinstance(meta, dict):
            la = meta.get("lastAccepted")
            if isinstance(la, dict):
                for k in ("insertedText", "textSnapshot"):
                    if isinstance(la.get(k), str):
                        lora_text_blobs.append(la[k])
    for blob in lora_text_blobs:
        for match in _LORA_RE.finditer(blob):
            lora_name = match.group(1)
            weight_str = match.group(2) or ""
            try:
                weight = float(weight_str.split(":")[0])
            except (ValueError, TypeError):
                weight = None
            # Inline tags don't carry an extension; folder_paths resolves the stem.
            _add(resolve_resource(lora_name, "loras", weight=weight, api_key=api_key))

    return resources


def _first_number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def extract_gen_params(prompt_graph):
    """
    Walk the prompt graph for generation parameters.

    Handles the core KSampler family, Flux guidance (BasicGuider/FluxGuidance +
    custom sampler chain), and the modular SamplerCustom chain. Returns a dict
    with any of: steps, sampler, scheduler, cfg, guidance, seed, width, height.
    Missing values are simply omitted.
    """
    params = {}

    def _set(key, value):
        if value is not None and key not in params:
            params[key] = value

    for _id, ctype, inputs in _iter_nodes(prompt_graph):
        ct = ctype.lower()
        if "ksampler" in ct or "samplercustom" in ct:
            _set("steps", _first_number(_get(inputs, "steps", prompt_graph)))
            _set("cfg", _first_number(_get(inputs, "cfg", prompt_graph)))
            sampler = _get(inputs, "sampler_name", prompt_graph)
            if isinstance(sampler, str):
                _set("sampler", sampler)
            scheduler = _get(inputs, "scheduler", prompt_graph)
            if isinstance(scheduler, str):
                _set("scheduler", scheduler)
            seed = _get(inputs, "seed", prompt_graph)
            if seed is None:
                seed = _get(inputs, "noise_seed", prompt_graph)
            _set("seed", _first_number(seed))
        elif ct == "basicscheduler":
            _set("steps", _first_number(_get(inputs, "steps", prompt_graph)))
            scheduler = _get(inputs, "scheduler", prompt_graph)
            if isinstance(scheduler, str):
                _set("scheduler", scheduler)
        elif ct == "ksamplerselect":
            sampler = _get(inputs, "sampler_name", prompt_graph)
            if isinstance(sampler, str):
                _set("sampler", sampler)
        elif "randomnoise" in ct or ct == "noise":
            _set("seed", _first_number(_get(inputs, "noise_seed", prompt_graph)))
        elif "fluxguidance" in ct:
            _set("guidance", _first_number(_get(inputs, "guidance", prompt_graph)))
        elif "emptylatent" in ct or "emptysd3latent" in ct or "latentimage" in ct:
            _set("width", _first_number(_get(inputs, "width", prompt_graph)))
            _set("height", _first_number(_get(inputs, "height", prompt_graph)))

    return params


def build_a1111_params(positive, negative, params, resources, width=None, height=None):
    """Assemble an A1111-style ``parameters`` string from extracted pieces."""
    positive = (positive or "").strip()
    negative = (negative or "").strip()

    lines = [positive, f"Negative prompt: {negative}"]

    parts = []
    if "steps" in params:
        parts.append(f"Steps: {params['steps']}")
    if "sampler" in params:
        parts.append(f"Sampler: {civitai_sampler_name(params['sampler'], params.get('scheduler', 'normal'))}")
    # Flux uses "guidance"; classic models use cfg.
    if "guidance" in params:
        parts.append(f"CFG scale: {params['guidance']}")
    elif "cfg" in params:
        parts.append(f"CFG scale: {params['cfg']}")
    if "seed" in params:
        parts.append(f"Seed: {params['seed']}")

    w = width if width is not None else params.get("width")
    h = height if height is not None else params.get("height")
    if w and h:
        parts.append(f"Size: {w}x{h}")

    parts.append("Version: ComfyUI")

    if resources:
        parts.append(f"Civitai resources: {json.dumps(resources, separators=(',', ':'))}")

    lines.append(", ".join(parts))
    return "\n".join(lines)


# Conditioning-mutating nodes that have no prompt text of their own; when a
# sampler's positive/negative traces into one of these, follow through to the
# encoder that produced the conditioning rather than stopping here.
_CONDITIONING_PASSTHROUGH = (
    "conditioningzeroout", "conditioningcombine", "conditioningconcat",
    "conditioningsettimesteprange", "conditioningsetarea", "conditioningaverage",
    "fluxguidance", "controlnetapply", "cfgguider", "basicguider",
)


def _trace_conditioning_text(ref, prompt_graph, _depth=0, _seen=None):
    """Follow a sampler conditioning link back to the prompt text feeding it.

    Anchors on the node that turns text into conditioning (the last step before
    the sampler) -- this is *any* encoder node with a ``text`` input, not just
    ``CLIPTextEncode`` (LoRA Manager's "Prompt" node, custom encoders, etc.).
    Walks through conditioning passthrough/combine nodes to reach it. Returns the
    resolved prompt string, or None if it can't be traced from the static graph.
    """
    if not isinstance(ref, list) or len(ref) != 2 or _depth > 24:
        return None
    node_id = ref[0]
    if _seen is None:
        _seen = set()
    if node_id in _seen:
        return None
    _seen.add(node_id)
    node = prompt_graph.get(node_id) if isinstance(prompt_graph, dict) else None
    if not isinstance(node, dict):
        return None
    ctype = node.get("class_type", "").lower()
    inputs = node.get("inputs", {}) or {}

    # An encoder: it carries a text input that becomes conditioning. Resolve it
    # (which follows text-builder chains via _resolve_link). Skip pure
    # passthrough nodes that merely transform existing conditioning.
    if "text" in inputs and not any(p in ctype for p in _CONDITIONING_PASSTHROUGH):
        text = _get(inputs, "text", prompt_graph)
        if isinstance(text, str) and text.strip():
            return text

    # Otherwise keep walking back along this node's conditioning links.
    for value in inputs.values():
        if isinstance(value, list):
            found = _trace_conditioning_text(value, prompt_graph, _depth + 1, _seen)
            if found is not None:
                return found
    return None


def _positive_negative_from_graph(prompt_graph):
    """Best-effort positive/negative prompt text feeding the sampler.

    Anchors on the sampler's ``positive``/``negative`` conditioning inputs and
    traces each back to the encoder that produced it -- the value right before
    it is encoded into conditioning. Works for CLIPTextEncode and custom
    encoders alike (LoRA Manager Prompt node, etc.). Returns empty strings when
    the wiring can't be resolved rather than risk mislabeling.
    """
    for _id, ctype, inputs in _iter_nodes(prompt_graph):
        ct = ctype.lower()
        if "ksampler" in ct or "samplercustom" in ct or "basicguider" in ct or "cfgguider" in ct:
            positive = _trace_conditioning_text(inputs.get("positive"), prompt_graph) or ""
            negative = _trace_conditioning_text(inputs.get("negative"), prompt_graph) or ""
            # If both resolve to the same text, the wiring is ambiguous -- skip.
            if positive and positive == negative:
                continue
            if positive or negative:
                return positive, negative

    # Fallback: a single resolvable CLIPTextEncode-style node, treated as positive.
    encodes = []
    for _id, ctype, inputs in _iter_nodes(prompt_graph):
        if "cliptextencode" in ctype.lower():
            text = _get(inputs, "text", prompt_graph)
            if isinstance(text, str) and text.strip():
                encodes.append(text)
    if len(encodes) == 1:
        return encodes[0], ""

    # Last resort: the prompt may be generated at runtime (LLM, wildcard, pipe)
    # and only persisted in a ShowText-style preview node's cached value.
    showtext = _showtext_prompt(prompt_graph)
    if showtext:
        return showtext, ""
    return "", ""


def _showtext_prompt(prompt_graph):
    """Return the cached text of a ShowText/preview node, if exactly one exists.

    ShowText-family nodes write their resolved runtime input back into the graph
    (text_0/text), which is often the only static trace of an LLM- or
    wildcard-generated prompt. Only used when exactly one such cached value is
    found, to avoid picking an unrelated display node.
    """
    found = []
    for _id, ctype, inputs in _iter_nodes(prompt_graph):
        ct = ctype.lower()
        if "showtext" in ct or "displaytext" in ct or "previewtext" in ct:
            for key in ("text_0", "text", "string"):
                val = inputs.get(key)
                if isinstance(val, str) and val.strip():
                    found.append(val)
                    break
    return found[0] if len(found) == 1 else ""


def auto_build_params(prompt_graph, width=None, height=None, api_key=None):
    """
    Reconstruct an A1111 ``parameters`` string from the prompt graph.
    Returns the string, or "" if nothing useful could be extracted.
    """
    if not isinstance(prompt_graph, dict) or not prompt_graph:
        return ""
    positive, negative = _positive_negative_from_graph(prompt_graph)
    params = extract_gen_params(prompt_graph)
    resources = extract_resources(prompt_graph, positive_text=positive, api_key=api_key)
    # Nothing meaningful to record -> signal "no metadata".
    if not positive and not params and not resources:
        return ""
    return build_a1111_params(positive, negative, params, resources, width=width, height=height)


def _decode_exif_user_comment(raw):
    """Decode an EXIF UserComment blob (A1111/Civitai store params here for JPEG).

    The tag is an 8-byte character-code prefix followed by the text. We handle the
    ``UNICODE`` (UTF-16) and ``ASCII`` prefixes A1111 uses, and fall back to a
    best-effort UTF-8/latin-1 decode for anything else. Returns a str or "".
    """
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, (bytes, bytearray)):
        return ""
    raw = bytes(raw)
    code = raw[:8]
    if code.startswith(b"UNICODE"):
        payload = raw[8:]
        for enc in ("utf-16-be", "utf-16-le", "utf-16"):
            try:
                return payload.decode(enc).rstrip("\x00")
            except Exception:
                continue
        return ""
    if code.startswith(b"ASCII"):
        return raw[8:].split(b"\x00", 1)[0].decode("ascii", "ignore")
    # Unknown/absent prefix: try utf-8 then latin-1.
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc).rstrip("\x00")
        except Exception:
            continue
    return ""


def extract_embedded_metadata(pil_img):
    """Pull generation metadata out of an already-opened image.

    Reads, in order of preference:
      * an A1111/Civitai ``parameters`` string (PNG text chunk or JPEG EXIF
        UserComment) -- the exact string the Civitai post nodes want;
      * the ComfyUI ``prompt`` graph (stock SaveImage embeds this, *not* a
        parameters string) so we can synthesize one via ``auto_build_params``;
      * the ComfyUI ``workflow`` graph, passed through for re-embedding and for
        drag-and-drop reproducibility.

    Returns a dict: ``a1111_params`` (str), ``prompt_graph`` (dict|None),
    ``workflow`` (dict|None). Any field may be empty/None.
    """
    result = {"a1111_params": "", "prompt_graph": None, "workflow": None}
    if pil_img is None:
        return result

    # PNG-style text chunks live in .info; JPEG params live in EXIF.
    info = getattr(pil_img, "info", {}) or {}

    params = info.get("parameters")
    if isinstance(params, bytes):
        params = params.decode("utf-8", "ignore")
    if isinstance(params, str) and params.strip():
        result["a1111_params"] = params.strip()

    # ComfyUI prompt / workflow JSON chunks (stock SaveImage).
    for key, dest in (("prompt", "prompt_graph"), ("workflow", "workflow")):
        blob = info.get(key)
        if isinstance(blob, (bytes, bytearray)):
            blob = bytes(blob).decode("utf-8", "ignore")
        if isinstance(blob, str) and blob.strip():
            try:
                result[dest] = json.loads(blob)
            except Exception:
                pass
        elif isinstance(blob, dict):
            result[dest] = blob

    # JPEG: parameters string lives in EXIF UserComment.
    if not result["a1111_params"]:
        try:
            exif = pil_img.getexif()
            exif_ifd = exif.get_ifd(IFD.Exif)
            raw = exif_ifd.get(_EXIF_USER_COMMENT)
            decoded = _decode_exif_user_comment(raw)
            if decoded.strip():
                result["a1111_params"] = decoded.strip()
        except Exception:
            pass

    # No explicit parameters string, but a prompt graph -> synthesize one so the
    # downstream post nodes still get full metadata (the stock-SaveImage case).
    if not result["a1111_params"] and isinstance(result["prompt_graph"], dict):
        synthesized = auto_build_params(
            result["prompt_graph"],
            width=getattr(pil_img, "width", None),
            height=getattr(pil_img, "height", None),
        )
        if synthesized:
            result["a1111_params"] = synthesized

    return result


# Maps an A1111 field label (lower-cased) to a (output_key, caster) pair.
def _to_int(v):
    try:
        return int(float(v))
    except (ValueError, TypeError):
        return 0


def _to_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def parse_a1111_params(params):
    """Parse an A1111 ``parameters`` string into individual fields.

    Inverse of :func:`build_a1111_params`. Handles the standard layout:

        <positive prompt>
        Negative prompt: <negative>
        Steps: N, Sampler: X, CFG scale: C, Seed: S, Size: WxH, ...

    The settings line is comma-separated ``Key: value`` pairs, but commas inside
    bracketed/JSON values (e.g. ``Civitai resources: [...]``) must not split a
    pair, so we only split on commas that are followed by a ``Key:`` token.

    Returns a dict with: prompt, negative_prompt, seed, steps, cfg_scale,
    sampler_name, width, height. Missing fields get empty/zero defaults.
    """
    out = {
        "prompt": "", "negative_prompt": "", "seed": 0, "steps": 0,
        "cfg_scale": 0.0, "sampler_name": "", "width": 0, "height": 0,
    }
    if not isinstance(params, str) or not params.strip():
        return out

    lines = params.split("\n")

    # Locate the "Negative prompt:" line and the settings line (last line that
    # looks like comma-separated Key: value pairs).
    neg_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Negative prompt:"):
            neg_idx = i
            break

    # Settings line: a trailing line containing "Key: value" pairs. A1111 puts it
    # on the last line; detect by the presence of a known key.
    settings_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if re.search(r"(?:^|,\s*)(Steps|Sampler|CFG scale|Seed|Size|Version):", lines[i]):
            settings_idx = i
            break

    # Positive prompt: everything before the negative line (or before settings).
    pos_end = neg_idx if neg_idx is not None else (settings_idx if settings_idx is not None else len(lines))
    out["prompt"] = "\n".join(lines[:pos_end]).strip()

    # Negative prompt: from the negative line up to the settings line.
    if neg_idx is not None:
        neg_end = settings_idx if settings_idx is not None and settings_idx > neg_idx else len(lines)
        neg_lines = lines[neg_idx:neg_end]
        neg_lines[0] = neg_lines[0][len("Negative prompt:"):]
        out["negative_prompt"] = "\n".join(neg_lines).strip()

    if settings_idx is None:
        return out

    settings = lines[settings_idx]
    # Split on commas that precede a "Key:" token, so JSON/bracketed values stay intact.
    pairs = re.split(r",\s*(?=[A-Za-z][\w ]*:\s)", settings)
    fields = {}
    for pair in pairs:
        if ":" not in pair:
            continue
        key, _, value = pair.partition(":")
        fields[key.strip().lower()] = value.strip()

    if "steps" in fields:
        out["steps"] = _to_int(fields["steps"])
    if "sampler" in fields:
        out["sampler_name"] = fields["sampler"]
    if "cfg scale" in fields:
        out["cfg_scale"] = _to_float(fields["cfg scale"])
    if "seed" in fields:
        out["seed"] = _to_int(fields["seed"])
    if "size" in fields:
        m = re.match(r"\s*(\d+)\s*x\s*(\d+)", fields["size"])
        if m:
            out["width"] = int(m.group(1))
            out["height"] = int(m.group(2))

    return out


def encode_image(pil_img, file_format="png", a1111_params="", embed_metadata=True,
                 embed_workflow=True, prompt=None, extra_pnginfo=None, jpg_quality=95,
                 workflow_override=None):
    """
    Encode a PIL image to bytes with the requested metadata embedded.

    Returns (image_bytes, content_type).

    PNG: ``parameters`` text chunk (+ ``prompt``/``workflow`` chunks when
    embed_workflow). PIL promotes non-latin-1 text to iTXt automatically, so
    Unicode prompts round-trip.
    JPG: ``parameters`` written to EXIF UserComment (UNICODE/UTF-16BE), matching
    the A1111/Civitai convention. Workflow chunks are PNG-only.

    ``workflow_override`` (a dict) takes precedence over the live ``prompt`` /
    ``extra_pnginfo`` graph: when a previously-saved image's workflow is supplied
    (e.g. from a Civitai metadata loader) it is embedded as the ``workflow``
    chunk so the post reproduces the *original* graph, not the current run's.
    """
    fmt = (file_format or "png").lower()
    buffered = io.BytesIO()

    if fmt in ("jpg", "jpeg"):
        if embed_workflow:
            print("[Civitai MCP] Note: workflow embedding is PNG-only; skipped for JPEG.")
        save_kwargs = {"format": "JPEG", "quality": int(jpg_quality)}
        if embed_metadata and a1111_params:
            exif = pil_img.getexif()
            exif_ifd = exif.get_ifd(IFD.Exif)
            exif_ifd[_EXIF_USER_COMMENT] = b"UNICODE\x00" + a1111_params.encode("utf-16-be")
            save_kwargs["exif"] = exif.tobytes()
        rgb = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
        rgb.save(buffered, **save_kwargs)
        return buffered.getvalue(), "image/jpeg"

    # PNG path
    pnginfo = PngInfo()
    if embed_metadata and a1111_params:
        pnginfo.add_text("parameters", a1111_params)
    if embed_workflow:
        if isinstance(workflow_override, dict) and workflow_override:
            # Wired workflow wins: embed the original graph from the loaded image.
            pnginfo.add_text("workflow", json.dumps(workflow_override))
        else:
            if prompt is not None:
                pnginfo.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
                for key, value in extra_pnginfo.items():
                    pnginfo.add_text(key, json.dumps(value))
    pil_img.save(buffered, format="PNG", pnginfo=pnginfo)
    return buffered.getvalue(), "image/png"
