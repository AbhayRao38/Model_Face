---
title: Mci Model Face
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

Model_Face
==========

This repository contains the facial-expression MCI/emotion model service.

Contents to copy into this repo before pushing:
- All files from `model_face/` including `face_api.py`, `train.py`, pretrained checkpoints in `pretrained/`.

Quick start (local):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m face_api
```

Deploy notes:
- Ensure `MODEL_PRETRAINED_PATH` env var is set to `pretrained/` or the model file path.
- For GPU, install CUDA-enabled PyTorch wheel before building the image.

Push example:

```bash
git remote add origin git@github.com:AbhayRao38/Model_Face.git
git push -u origin main
```
