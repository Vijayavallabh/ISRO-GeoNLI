# Model backends are auto-registered via imports
from geonli.models.base import DummyVLM, DummySegmenter
from geonli.models.huggingface import HuggingFaceVLM
from geonli.models.huggingface_sam import HuggingFaceSAM
try:
    from geonli.models.huggingface_sam3 import HuggingFaceSAM3
except Exception:
    pass
