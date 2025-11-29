"""
Image captioning task implementation.
"""


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
        Generate detailed caption using two-stage approach.
        
        Args:
            image: PIL Image
            instruction: User instruction for captioning
            
        Returns:
            Final compressed caption string
        """
        print(f"--- Task: Captioning ---")
        
        # Stage 1: Generate comprehensive draft
        draft_prompt = (
            f"{instruction}\n"
            "Draft a comprehensive description listing all visible objects, "
            "their counts, colors, and relative positions. Be verbose."
        )
        long_caption = self.vlm.query(image, draft_prompt, max_tokens=400)
        
        # Stage 2: Compress to target length (~60 words)
        compress_prompt = (
            f"Here is a detailed description of the image:\n'{long_caption}'\n\n"
            f"User Instruction: {instruction}\n"
            "Task: Summarize this description into a single, high-density caption.\n"
            "Constraints:\n"
            "1. Target length: Approximately 60 words.\n"
            "2. Do NOT exceed 80 words.\n"
            "3. You must use complete, grammatical sentences.\n"
            "4. Retain ALL specific details and relative positions.\n"
            "5. Remove 'fluff'"
        )
        
        final_caption = self.vlm.query(image, compress_prompt, max_tokens=128)
        
        print(f"-> Draft Length: {len(long_caption.split())} words")
        print(f"-> Final Length: {len(final_caption.split())} words")
        
        return final_caption
