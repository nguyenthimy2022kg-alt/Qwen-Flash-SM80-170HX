# 固定已核对的公开上游；不使用本机实验镜像作为发布依赖。
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
ARG DEBIAN_FRONTEND=noninteractive
ARG CUFILE_VERSION=1.15.1.6-1
ARG CUDA_DRIVER_DEV_VERSION=13.0.96-1
RUN apt-get update -o Dir::Etc::sourcelist="sources.list.d/cuda.list" -o Dir::Etc::sourceparts="-" && apt-get install -y --no-install-recommends \
    cuda-driver-dev-13-0=${CUDA_DRIVER_DEV_VERSION} \
    libcufile-13-0=${CUFILE_VERSION} libcufile-dev-13-0=${CUFILE_VERSION} \
    && rm -rf /var/lib/apt/lists/*
# 不自动升级基础镜像内的 PyTorch、Triton 或 NumPy。
ARG PIP_INDEX_URL=https://pypi.org/simple
COPY requirements-kernels.txt /tmp/requirements-kernels.txt
RUN python3 -m pip install --disable-pip-version-check --no-deps --require-hashes --index-url ${PIP_INDEX_URL} -r /tmp/requirements-kernels.txt
COPY . /opt/qwen-flash-sm80
WORKDIR /opt/qwen-flash-sm80
RUN CUDA_ROOT=/usr/local/cuda-13.0 \
    CUDA_HEADER_DIR=/usr/local/cuda-13.0/targets/x86_64-linux/include \
    LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs \
    bash scripts/build-ple-gds-reader.sh \
    && python3 scripts/apply-overlay.py --target /usr/local/lib/python3.12/dist-packages \
    && python3 -m compileall -q src \
    && ldconfig
ENV PYTHONPATH=/usr/local/lib/python3.12/dist-packages
ENTRYPOINT []
CMD ["python3", "-m", "vllm.entrypoints.openai.api_server", "--help"]
