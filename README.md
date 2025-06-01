# Edge-AI-final

## Clone Repository

```bash
git clone https://github.com/fakraze/Edge-AI-final.git
cd Edge-AI-final
```

---

## Environment Setup

### 1. Create virtual environment

```bash
python3 -m venv hqq-env
source hqq-env/bin/activate
```

### 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
--extra-index-url https://download.pytorch.org/whl/cu121
```

### 3. Install other dependencies

```bash
pip install -r requirements.txt
```

---

## Hugging Face Authentication

```bash
huggingface-cli login
```

Then paste your Hugging Face token when prompted.

---

## Run Inference

```bash
python result.py
```
