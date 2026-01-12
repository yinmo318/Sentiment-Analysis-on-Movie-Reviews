# Sentiment Analysis on Movie Reviews (Kaggle)

This repo contains code for the Kaggle competition “Sentiment Analysis on Movie Reviews”.

## Setup (Windows / PowerShell)

```powershell
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1

# (GPU) PyTorch
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
