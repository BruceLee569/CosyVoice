# AutoDL 服务器部署指南

## 问题诊断结果

### 根本原因

在 AutoDL 环境中，直接运行 `python fast_server.py` 时，`import tensorrt_llm` 会卡住。这不是代码问题，而是 **TensorRT-LLM 在非 MPI 环境下导入时的已知问题**。

**核心原因：**
- TensorRT-LLM 在 `import` 阶段会尝试初始化 MPI 环境
- 即使设置了 PMI/MPI 环境变量，某些 MPI 实现仍会尝试启动后台进程
- 在 AutoDL 的 OpenMPI 环境中，这个后台进程启动会无限等待 → 卡死

### 为什么本地 WSL2 能跑通？

- **本地 WSL2**：OpenMPI 配置完整，后台进程能正常启动（即使是单进程模式）
- **AutoDL**：OpenMPI 配置不同，后台进程启动机制有差异

---

## ✅ 解决方案：使用 mpirun 启动脚本

### 在 AutoDL 上启动服务

**方法1：使用 MPI 启动脚本（推荐）**

```bash
cd /root/autodl-tmp/CosyVoice
source .venv/bin/activate

# 给脚本添加执行权限
chmod +x start_server_mpi.sh

# 使用 mpirun 启动（自动规避 MPI 初始化问题）
./start_server_mpi.sh
```

**方法2：手动使用 mpirun**

```bash
cd /root/autodl-tmp/CosyVoice
source .venv/bin/activate

mpirun -n 1 --allow-run-as-root --bind-to none python fast_server.py
```

### 在本地 WSL2 上启动服务

本地环境可以直接运行（不需要 mpirun）：

```bash
cd /home/cz/ai/CosyVoice
python fast_server.py
```

---

## 🔧 技术原理

### mpirun 的作用

通过 `mpirun -n 1` 启动程序后：

1. **MPI 环境正确初始化**：mpirun 会启动 PMI 服务和必要的后台进程
2. **环境变量自动设置**：`OMPI_COMM_WORLD_RANK`、`PMI_RANK` 等会被正确设置
3. **`import tensorrt_llm` 不再卡住**：TensorRT-LLM 检测到已在 MPI 环境中，不会再尝试初始化

### 为什么单进程也需要 mpirun？

- TensorRT-LLM 的设计假设：总是在 MPI 环境中运行（即使单 GPU）
- 直接 `python` 启动：缺少 MPI 运行时环境 → 导入卡住
- `mpirun -n 1` 启动：提供完整的 MPI 运行时 → 正常运行

---

## 📋 预期输出

### 正常启动日志

```
========================================================================
  FastCosyVoice TTS Server - MPI 启动脚本（AutoDL 优化版）
========================================================================
✓ 使用 mpirun 单进程模式启动（规避 AutoDL MPI 初始化问题）

============================================================
启动 FastCosyVoice TTS Server
模型目录: pretrained_models/CosyVoice2-0.5B
TensorRT-LLM 引擎目录: pretrained_models/cosyvoice2_llm/trt_engines_bfloat16
TensorRT-LLM Tokenizer: pretrained_models/cosyvoice2_llm
============================================================
...
2025-12-12 XX:XX:XX.XXX INFO 正在导入 TensorRT-LLM...
2025-12-12 XX:XX:XX.XXX INFO ✅ TensorRT-LLM 导入成功
2025-12-12 XX:XX:XX.XXX INFO 使用 MPI rank=0
...
🚀 FastCosyVoice TTS Server 启动在端口 6008
```

### 如果还是卡住

如果使用 `start_server_mpi.sh` 后仍然卡住，请检查：

1. **OpenMPI 是否正确安装**：
   ```bash
   which mpirun
   mpirun --version
   ```

2. **检查权限**：
   ```bash
   ls -la start_server_mpi.sh
   # 应该显示 -rwxr-xr-x（可执行）
   ```

3. **手动测试 mpirun**：
   ```bash
   mpirun -n 1 --allow-run-as-root python -c "import tensorrt_llm; print('OK')"
   ```

---

## 🚨 常见问题

### Q1: 为什么不能直接 `python fast_server.py`？

**A:** 在 AutoDL 环境中，TensorRT-LLM 的 MPI 初始化机制与环境配置不兼容，必须通过 mpirun 提供完整的 MPI 运行时。

### Q2: mpirun 会影响性能吗？

**A:** 不会。`mpirun -n 1` 只是提供 MPI 运行时环境，实际只运行一个进程，性能与直接运行完全相同。

### Q3: 能否彻底禁用 MPI？

**A:** 不能。TensorRT-LLM 的 C++ 底层代码强制依赖 MPI，无法在编译后禁用。

---

## 📦 部署清单

### AutoDL 环境

- [x] 安装 OpenMPI：`apt-get install openmpi-bin openmpi-common libopenmpi-dev`
- [x] 使用启动脚本：`./start_server_mpi.sh`
- [x] 验证服务：访问 `http://localhost:6008`

### 本地 WSL2 环境

- [x] 直接运行：`python fast_server.py`
- [x] 验证服务：访问 `http://localhost:6008`

---

## 🔗 相关资源

- [TensorRT-LLM MPI Issue #116](https://github.com/NVIDIA/TensorRT-LLM/issues/116)
- [TensorRT-LLM MPI4PY Issue #2286](https://github.com/NVIDIA/TensorRT-LLM/issues/2286)
- [OpenMPI Documentation](https://www.open-mpi.org/doc/)
