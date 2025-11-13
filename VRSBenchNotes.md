colab notebook for VRSBench Evaluation --> https://colab.research.google.com/drive/1HZZk01GzVzcbM-h6n8LFdpFzz16Ga-gh?usp=sharing

- the VRSBench repo is horrible, like it is very unorganised and unclear, and their evalution notebooks don't work... so use this notebook to directly run inference on whatever model from huggingface
- apparently there is some new metric ISRO is going to use to judge captions called BleuRT, I have included that as well along with the ones they used in the paper
- captioning eval is sorted for now

TODO: 
- the run_evaluations function had some hardcoded-ness for Qwen models, so will need to make that model agnostic by defning a generate function
- add VQA (simmilar to captioning, just need to change system prompt and metrics by which we evaluate the answer quality)
- grounding (OBBs --> kinda lost here, could use some help) 
- inference on ScoreRS's model on JarvisLabs
