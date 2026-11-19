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

## Credits & Acknowledgements

### Microsoft Research, Dense Landmark-Based 3D Face Reconstruction

Some of the landmark weighting and reconstruction concepts in this project were inspired by Microsoft's research paper:

**[3D Face Reconstruction with Dense Landmarks (ECCV 2022)](https://arxiv.org/abs/2204.02776)** — Erroll Wood et al.

The paper explores dense facial landmarks, uncertainty-aware landmark fitting, and 3D face reconstruction. Its ideas informed parts of the landmark reliability and geometry regularization approach developed in this project.

> *This project implements its own reconstruction pipeline and does not use or redistribute Microsoft's trained landmark detection model or implementation.*

### Google — MediaPipe

Facial landmark detection is powered by **[MediaPipe](https://github.com/google-ai-edge/mediapipe)**, an open-source framework developed by Google.

- **Documentation:** [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- **License:** [Apache License 2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)