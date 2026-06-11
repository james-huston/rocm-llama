# syntax=docker/dockerfile:1.6
#
# rocm-llama — ROCm/HIP-accelerated llama.cpp + llama-swap for AMD GPUs
# (RX 7900 XTX, RX 7800 XT, RX 6900 XT, MI300X, etc.). Self-contained: no host
# ROCm toolkit install needed beyond the kernel driver.
#
# Build:  docker compose build
# Run:    docker compose up -d
# Docs:   https://github.com/james-huston/rocm-llama

ARG ROCM_IMAGE_TAG=7.2.1-complete
FROM rocm/dev-ubuntu-24.04:${ROCM_IMAGE_TAG}

# -----------------------------------------------------------------------------
# System packages
#
# ROCm dev image already ships hipcc, clang, rocminfo, and the ROCm libraries.
# We add cmake, ninja, git, and the usual build tools.
# -----------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git cmake ninja-build build-essential pkg-config \
        ca-certificates curl wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# The ROCm dev image ships its HIP-aware clang at /opt/rocm/llvm/bin but does
# NOT put it on PATH, so the build steps below that ask for clang/clang++ would
# otherwise fail to configure. Put ROCm's LLVM toolchain on PATH (this is the
# correct compiler for the HIP backend; plain apt clang would not be).
ENV PATH="/opt/rocm/llvm/bin:${PATH}"

# llama.cpp/sd.cpp link against ROCm clang's OpenMP runtime (libomp.so), which
# lives in ROCm's LLVM lib dir — NOT the default /opt/rocm/lib that ldconfig
# already knows about. Without this the binaries fail at runtime with
# "libomp.so: cannot open shared object file" (exit 127). Register the path so
# the dynamic loader finds it.
RUN echo "/opt/rocm/llvm/lib" > /etc/ld.so.conf.d/rocm-llvm.conf && ldconfig

# -----------------------------------------------------------------------------
# Build llama.cpp with HIP/ROCm backend
#
# LLAMA_CPP_REF defaults to 'master' (always latest). Pin to a specific tag
# like 'b9409' for reproducible builds: docker compose build --build-arg
# LLAMA_CPP_REF=b9409
#
# GPU_TARGETS sets the AMDGPU architecture to compile for (gfx1100 = RX 7900 XTX,
# gfx1030 = RX 6800, gfx942 = MI300X). Omit to build for all detected targets.
#
# -DGGML_HIP=ON              enables the HIP/ROCm backend
# -DGGML_HIP_ROCWMMA_FATTN=ON enables flash attention on RDNA3+/CDNA (requires rocWMMA)
#
# Native static build: GGML_NATIVE (default ON) tunes the single CPU backend for
# the build host. We deliberately do NOT use GGML_BACKEND_DL — it would force
# GGML_CPU_ALL_VARIANTS (compiling ~15 x86 microarch variants incl. the very slow
# AMX/AVX-512 ones this host can't even use), ballooning build time for no benefit
# on a single-arch, single-host image. The image is already pinned to one
# GPU_TARGETS, so dynamic backend loading buys nothing here.
# -----------------------------------------------------------------------------
ARG LLAMA_CPP_REF=master
ARG GPU_TARGETS=gfx1100
# rocWMMA flash-attention needs WMMA matrix instructions that exist only on
# RDNA3+/CDNA. RDNA2 (gfx103x — e.g. RX 6650 XT) lacks them, so building with
# this ON for an RDNA2 target fails to compile the FA kernels. Pass OFF for
# RDNA2: docker compose build --build-arg GGML_HIP_ROCWMMA_FATTN=OFF
ARG GGML_HIP_ROCWMMA_FATTN=ON
RUN git clone --depth 1 --branch ${LLAMA_CPP_REF} \
        https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp && \
    cd /tmp/llama.cpp && \
    cmake -B build -G Ninja \
        -DGGML_HIP=ON \
        -DGPU_TARGETS=${GPU_TARGETS} \
        -DGGML_HIP_ROCWMMA_FATTN=${GGML_HIP_ROCWMMA_FATTN} \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j $(nproc) && \
    cmake --install build --prefix=/opt/llama-cpp && \
    echo "/opt/llama-cpp/lib" > /etc/ld.so.conf.d/llama-cpp.conf && \
    ldconfig && \
    rm -rf /tmp/llama.cpp

# -----------------------------------------------------------------------------
# Build stable-diffusion.cpp server (sd-server) with HIPBLAS — on-demand image
# generation, swapped in by llama-swap like any model. sd-server serves OpenAI
# /v1/images/generations (+ SDAPI), which llama-swap proxies.
#
# GPU_TARGETS must match the llama.cpp target.
# -----------------------------------------------------------------------------
ARG SD_CPP_REF=master
ARG SD_GPU_TARGETS=gfx1100
RUN git clone --recurse-submodules --shallow-submodules --depth 1 --branch ${SD_CPP_REF} \
        https://github.com/leejet/stable-diffusion.cpp.git /tmp/sd.cpp && \
    cd /tmp/sd.cpp && \
    cmake -B build -G Ninja \
        -DSD_HIPBLAS=ON \
        -DGPU_TARGETS=${SD_GPU_TARGETS} \
        -DAMDGPU_TARGETS=${SD_GPU_TARGETS} \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release --target sd-server -j $(nproc) && \
    mkdir -p /opt/sd-cpp/bin /opt/sd-cpp/lib && \
    cp build/bin/sd-server /opt/sd-cpp/bin/ && \
    find build -name '*.so*' -exec cp -a {} /opt/sd-cpp/lib/ \; && \
    echo "/opt/sd-cpp/lib" > /etc/ld.so.conf.d/sd-cpp.conf && \
    ldconfig && \
    rm -rf /tmp/sd.cpp

# -----------------------------------------------------------------------------
# Install llama-swap
#
# Override version at build time if needed:
#   docker compose build --build-arg LLAMA_SWAP_VERSION=200
# -----------------------------------------------------------------------------
ARG LLAMA_SWAP_VERSION=217
RUN wget -qO /tmp/llama-swap.tar.gz \
        "https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION}_linux_amd64.tar.gz" && \
    mkdir -p /opt/llama-swap/bin && \
    tar -C /opt/llama-swap/bin -xzf /tmp/llama-swap.tar.gz llama-swap && \
    rm /tmp/llama-swap.tar.gz && \
    /opt/llama-swap/bin/llama-swap --version

# -----------------------------------------------------------------------------
# Runtime configuration
# -----------------------------------------------------------------------------
ENV PATH=/opt/llama-swap/bin:/opt/llama-cpp/bin:/opt/sd-cpp/bin:${PATH}

# Select which GPU(s) to use (same semantics as CUDA_VISIBLE_DEVICES).
# HIP_VISIBLE_DEVICES=0 selects the first ROCm GPU.
ENV HIP_VISIBLE_DEVICES=0

# Override GPU detection for unsupported cards (Linux only).
# e.g. HSA_OVERRIDE_GFX_VERSION=10.3.0 for RDNA2 cards not natively supported.
# ENV HSA_OVERRIDE_GFX_VERSION=

# llama-swap listen port — matches Ollama's default so LiteLLM / clients that
# expected Ollama on 11434 don't need reconfiguration.
EXPOSE 11434

# Default: run llama-swap with hot-reload on config changes.
# Config file is expected to be bind-mounted at /config/llama-swap.yaml.
CMD ["/opt/llama-swap/bin/llama-swap", \
     "--config", "/config/llama-swap.yaml", \
     "--listen", "0.0.0.0:11434", \
     "--watch-config"]
