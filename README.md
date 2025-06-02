## Environment Setup

### Get Started (For students just start using GPU Servers)
```
sudo apt update
sudo apt install build-essential
sudo apt install python3-pip
sudo apt install python3-venv
sudo apt install git
```
### Install Package
```
pip3 install huggingface-hub[cli]
pip3 install transformers==4.50.3
pip3 install torch==2.6.0 torchvision torchaudio
pip3 install timm==1.0.15
pip3 install datasets==3.5.0
pip3 install accelerate==1.6.0
pip3 install gemlite==0.4.4
pip3 install hqq==0.2.5
pip3 install triton==3.2.0
```
```
pip install "sglang[all]"
pip install gptqmodel --no-build-isolation -v
ip install auto-gptq
pip install optimum
pip install vllm==0.8.2
```
### Inference
```
huggingface-cli login
python3 main.py
```