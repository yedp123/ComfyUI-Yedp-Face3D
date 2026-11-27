# ComfyUI-Yedp-Face3D
experimental MediaPipe-based 3D face reconstruction for AI image conditioning


https://github.com/user-attachments/assets/c6afc10e-4334-4708-9e84-79d5ad033878

<img width="872" height="825" alt="45d1b1d66ba28dfcece68e79700e9cd8" src="https://github.com/user-attachments/assets/b8393f3b-e6d8-4cc3-aaf0-548e6545270d" />


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

### Silhouette-Guided 3D Face Reconstruction

The constrained side-view silhouette fitting approach in this project draws conceptual inspiration from silhouette- and segmentation-guided 3D face reconstruction research, including:

**[3DDFA-V3: Face Reconstruction with Part-level Representation and Geometric Guidance (CVPR 2024)](https://arxiv.org/abs/2312.00311)** — Zidu Wang et al.

> *This project uses its own sub-pixel scanline silhouette refinement driven by MediaPipe segmentation masks and does not include or redistribute 3DDFA-V3 models or code.*

### Classical Computer Vision & Graphics Foundations

The multi-view UV texture baking, color harmonization, and eyeball shading stages build on established computer graphics and image-processing techniques:

- **Multi-Band (Laplacian Pyramid) Seam Blending:** Based on **[A Multiresolution Spline With Application to Image Mosaics (ACM TOG 1983)](https://persci.mit.edu/pub_pdfs/spline83.pdf)** by Peter J. Burt and Edward H. Adelson, implemented via OpenCV CPU pyramids with frequency-split detail preservation.
- **Overlap CIE Lab Color Harmonization:** Inspired by **[Color Transfer between Images (IEEE CG&A 2001)](https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf)** by Erik Reinhard et al., computed strictly across shared, non-hair skin regions.
- **Three.js & OpenCV:** Powered by **[Three.js](https://threejs.org/)** (MIT License) for real-time WebGL rendering, harmonic cage skinning, and raycast eyeball occlusion blending, and **[OpenCV](https://opencv.org/)** (Apache License 2.0) for CPU UV projection and image processing.
