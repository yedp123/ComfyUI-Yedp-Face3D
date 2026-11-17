# ComfyUI-Yedp-Face3D
experimental MediaPipe-based 3D face reconstruction for AI image conditioning

https://github.com/user-attachments/assets/b200bc27-735b-46c8-8632-e989ed5bb4eb

This is not photogrammetry and doesn't produce a production-ready 3D head. It uses MediaPipe's 478 facial landmarks to create an approximate, controllable 3D representation of a reference face.  
The original motivation was to explore whether this lightweight geometric representation could provide useful shape guidance for AI image generation through depth, normal and other rendered passes.

# Known limitations

- approximate geometry
- face only
- no full head reconstruction
- not a replacement for LoRA
- generation-model dependent
- experimental ControlNet workflows
