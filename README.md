# ⚡ Optimized CLIP Implementations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

> **A research repository dedicated to building and optimizing Vision-Language Models (VLMs) from scratch. The primary goal is to achieve real-time inference speeds on consumer hardware without sacrificing semantic accuracy.**

Developed by **Muhammed Köse** as an **Academic Research Project**.

---

## 📂 Project Structure & Versions

Currently, this repository contains the **v1.0** implementation of the Custom CLIP architecture.

| Version | Architecture | Optimization | Key Result | Link |
| :--- | :--- | :--- | :--- | :--- |
| **v1.0** | ViT-B/16 | FP16 + JIT Compile | **🚀 2.46x Speedup** | [**View Source Code**](./ClipVitB16-v1.0) |

> *Click on the link above to access the source code, training details, and inference scripts.*

---

## 🎯 Research Goal

Standard implementations of large models (like OpenAI's CLIP) prioritize flexibility and training stability over **inference latency**. This project aims to bridge the gap between heavy academic models and **real-time edge deployment**.

**Core Objectives:**
1.  **Re-implementation:** Coding Vision Transformers (ViT) and Text Encoders from scratch to understand every layer.
2.  **Optimization:** Utilizing `torch.compile` and Mixed Precision (FP16) to maximize GPU throughput.
3.  **Benchmarking:** Rigorous comparison against industry standards (OpenAI) on local hardware (e.g., RTX 3050 Ti).

---

## 🏆 Current Highlights (v1.0)

Our latest stable release (**v1.0**) has achieved significant performance gains:

* **Latency:** Reduced from **52ms** to **21ms** per image.
* **Throughput:** Increased from **19 FPS** to **47+ FPS**.
* **Accuracy:** maintained **97.7%** Zero-Shot confidence on COCO dataset samples.

For detailed benchmarks and usage instructions, please navigate to the [v1.0 Directory](./ClipVitB16-v1.0).

---

## 📜 Citation

If you use the implementations in this repository, please cite the project as follows:

```bibtex
@misc{kose2026customclip,
  author = {Köse, Muhammed},
  title = {Custom CLIP: Optimized Vision-Language Model},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{[https://github.com/MuhammedKsee/custom-clip-vit-b-coco](https://github.com/MuhammedKsee/custom-clip-vit-b-coco)}}
}
```
## 📄 License
This repository is licensed under the MIT License.
