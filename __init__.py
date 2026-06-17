from .civitai_nodes import (
    CivitaiPostImage,
    CivitaiCreatePost,
    CivitaiGetCurrentChallenge,
    CivitaiGetModelMetadata,
    CivitaiGetImageMetadata,
    CivitaiAccountStatus,
    CivitaiGetImage
)
from .civitai_loaders import (
    CivitaiLoadImageWithMetadata,
    CivitaiLoadImagesFromDir,
    CivitaiParseA1111Params,
)

NODE_CLASS_MAPPINGS = {
    "CivitaiPostImage": CivitaiPostImage,
    "CivitaiCreatePost": CivitaiCreatePost,
    "CivitaiGetCurrentChallenge": CivitaiGetCurrentChallenge,
    "CivitaiGetModelMetadata": CivitaiGetModelMetadata,
    "CivitaiGetImageMetadata": CivitaiGetImageMetadata,
    "CivitaiAccountStatus": CivitaiAccountStatus,
    "CivitaiGetImage": CivitaiGetImage,
    "CivitaiLoadImageWithMetadata": CivitaiLoadImageWithMetadata,
    "CivitaiLoadImagesFromDir": CivitaiLoadImagesFromDir,
    "CivitaiParseA1111Params": CivitaiParseA1111Params,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CivitaiPostImage": "Civitai Post Image",
    "CivitaiCreatePost": "Civitai Create Post",
    "CivitaiGetCurrentChallenge": "Civitai Get Current Challenge",
    "CivitaiGetModelMetadata": "Civitai Get Model Metadata",
    "CivitaiGetImageMetadata": "Civitai Get Image Metadata",
    "CivitaiAccountStatus": "Civitai Account Status",
    "CivitaiGetImage": "Civitai Get Image",
    "CivitaiLoadImageWithMetadata": "Civitai Load Image (with Metadata)",
    "CivitaiLoadImagesFromDir": "Civitai Load Images from Dir (with Metadata)",
    "CivitaiParseA1111Params": "Civitai Parse A1111 Params",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
