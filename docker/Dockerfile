FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
LABEL maintainer="dangnh0611@gmail.com"

RUN apt update
RUN apt install -y zsh
RUN apt install -y git
RUN DEBIAN_FRONTEND=noninteractive apt install -y build-essential make cmake
RUN apt update
RUN apt install -y ffmpeg

# Pip dependencies
RUN pip install --upgrade pip
RUN pip install hydra-core --upgrade
RUN pip install torchvision lightning transformers scikit-learn scipy seaborn matplotlib numba yacs pyyaml fvcore sympy Cython cython_bbox albumentations audiomentations wandb datasets lap catboost polars
RUN pip install git+https://github.com/rwightman/pytorch-image-models.git
RUN pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
RUN pip install onnxoptimizer onnxsim onnxruntime-gpu tensorrt pycuda

# Apex
WORKDIR /workspace/lib/
RUN git clone https://github.com/NVIDIA/apex
WORKDIR /workspace/lib/apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
WORKDIR /workspace


# # torch2trt
# WORKDIR /workspace/lib/
# RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt
# # COPY torch2trt torch2trt
# WORKDIR /workspace/lib/torch2trt
# RUN python3 setup.py install


# Competition-related packages
RUN pip install --pre deepchem
RUN pip install --upgrade 'rdkit[all]' atomInSmiles SmilesPE deepsmiles selfies scikit-fingerprints py3Dmol cdpkit mordredcommunity
RUN pip install mapply torch_geometric 'causal-conv1d>=1.4.0' mamba-ssm shap

# for visualization of CatBoost training
# RUN pip install notebook==6.1.5 ipywidgets
# RUN jupyter nbextension enable --py widgetsnbextension

WORKDIR /workspace/
CMD ["/usr/bin/zsh"]
