FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-drivers \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    make \
    git \
    gcc \
    clang \
    curl \
    libcurl4-openssl-dev \
    python3 \
    python3-pip \
    wget \
    xz-utils \
    libboost-all-dev \
    libeigen3-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/zig && \
    curl -L https://ziglang.org/builds/zig-x86_64-linux-0.15.0-dev.936+fc2c1883b.tar.xz | tar -xJ -C /opt/zig --strip-components=1

ENV PATH="/opt/zig:${PATH}"

RUN zig version && \
    gcc --version && \
    nvcc --version

WORKDIR /zLLMChat

COPY build_with_cuda.sh ./
COPY build_without_cuda.sh ./

#RUN mkdir models && chmod +x build_without_cuda.sh && ./build_without_cuda.sh
RUN mkdir models && chmod +x build_with_cuda.sh && ./build_with_cuda.sh

COPY src/ ./src/
COPY build.zig ./
COPY build.zig.zon ./

CMD ["/bin/bash"]