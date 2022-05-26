# derived from Microsoft Python dev container (https://github.com/microsoft/vscode-dev-containers/tree/main/containers/python-3)
FROM mcr.microsoft.com/vscode/devcontainers/python:3.7

# Colab versions @2022-05-24
# Python:                 3.7.13
# NumPy:      numpy       1.21.6
# PyTorch:    torch       1.11.0+cu113
# Torchaudio: torchaudio  0.11.0+cu113

# PyTorch on CPU
RUN pip3 --disable-pip-version-check --no-cache-dir install numpy==1.21.6
RUN pip3 --disable-pip-version-check --no-cache-dir install torch==1.11.0+cpu torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# package dependency manager `poetry` without environment isolation (It is a job of Docker.)
ENV POETRY_VIRTUALENVS_CREATE=false
RUN curl -sSL https://install.python-poetry.org | python3 -