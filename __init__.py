from .civitai_nodes import (
    CivitaiPostImage,
    CivitaiCreatePost,
    CivitaiGetCurrentChallenge,
    CivitaiGetModelMetadata,
    CivitaiGetImageMetadata,
    CivitaiAccountStatus,
    CivitaiGetImage
)

NODE_CLASS_MAPPINGS = {
    "CivitaiPostImage": CivitaiPostImage,
    "CivitaiCreatePost": CivitaiCreatePost,
    "CivitaiGetCurrentChallenge": CivitaiGetCurrentChallenge,
    "CivitaiGetModelMetadata": CivitaiGetModelMetadata,
    "CivitaiGetImageMetadata": CivitaiGetImageMetadata,
    "CivitaiAccountStatus": CivitaiAccountStatus,
    "CivitaiGetImage": CivitaiGetImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitaiPostImage": "Civitai Post Image",
    "CivitaiCreatePost": "Civitai Create Post",
    "CivitaiGetCurrentChallenge": "Civitai Get Current Challenge",
    "CivitaiGetModelMetadata": "Civitai Get Model Metadata",
    "CivitaiGetImageMetadata": "Civitai Get Image Metadata",
    "CivitaiAccountStatus": "Civitai Account Status",
    "CivitaiGetImage": "Civitai Get Image"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']


