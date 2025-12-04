"""
Image captioning task implementation.
"""
import logging
logger = logging.getLogger(__name__)

class CaptioningTask:
    """Handles image captioning with generate-then-compress strategy."""
    
    def __init__(self, vlm_interface):
        """
        Args:
            vlm_interface: VLMInterface instance
        """
        self.vlm = vlm_interface

    def generate_caption(self, image, instruction):
        """
        Generate a detailed but concise caption using a single VLM call
        guided by a system prompt.

        Args:
            image: PIL Image
            instruction: User instruction for captioning

        Returns:
            Final caption string
        """
        SYSTEM_PROMPT_CAPTION = """
        You are a vision-language assistant specialized in factual image captioning.
        
        Your task is to generate a clear, well-structured caption of approximately 50–150 words that describes all clearly visible content in the image. 
        The caption should be concise, crisp, and focused, avoiding unnecessary or repetitive details while covering all important visual elements.
        
        ## Important Guidelines for Image Captioning:
        1. Begin with a brief high-level overview of the scene, then describe specific, unambiguous visible details.
        2. Describe only what is directly observable in the image; do not speculate, infer intent, or add uncertain information.
        3. Include prominent objects and structural elements such as vehicles, buildings, roads, natural features, or infrastructure when clearly visible.
        4. Describe relevant visual attributes such as color, shape, position, relative location, orientation, and arrangement without introducing unsupported measurements.
        5. When applicable, note clear structural layouts or patterns (e.g., road networks, building clusters, field arrangements).
        6. Do not include imagined, inferred, or non-visual information such as weather, time of day, object purpose, or motion unless explicitly visible.
        7. Ensure the caption remains factual, internally consistent, and to the point, with no extraneous explanations.
        
        """
        
        final_caption = self.vlm.query(image, instruction, SYSTEM_PROMPT_CAPTION, max_tokens=256)

        return final_caption
