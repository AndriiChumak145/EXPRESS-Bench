# EXPRESS-Bench Installation Status (RTX 5090 / Blackwell)

This is the final status log of the installation and validation of the conda environment `fine-eqa` on the RTX 5090 GPU (`sm_120`).

## Final Status: SUCCESS

The environment has been fully created, resolved, and verified to be working on the RTX 5090 GPU without modifying any project source code.

### 1. Environment & Package Selection
*   **Python Version**: `3.11.15`.
    *   *Why*: Precompiled CUDA 12.8 / 13.0 PyTorch packages and the required nightly `habitat-sim` builds are only available for Python 3.10+.
*   **PyTorch**: `2.10.0+cu128` (installed automatically from NGC/PyPI dependencies).
    *   *Why*: Native `cu128` support is required to run on the RTX 5090 Blackwell GPU.
*   **Habitat-Sim**: `0.3.3` (nightly).
    *   *Why*: Precompiled for Python 3.11 on the `aihabitat-nightly` channel, avoiding a lengthy and error-prone local C++ compilation.

### 2. Resolved Setup Challenges (No Source Code Changes)
To run the environment successfully, we solved two major compatibility issues purely through package installation and runtime configurations:

1.  **Magnum / Numba ABI Conflict (Segmentation Fault)**:
    *   *Issue*: Precompiled mypyc-compiled binary wheels of the package `charset-normalizer` (installed by default for `requests`/`timm`) cause a segmentation fault when loaded alongside `habitat-sim`'s shared graphics libraries.
    *   *Solution*: Reinstalled `charset-normalizer` from source using `--no-binary :all:`. This forces it to run as a pure-Python wheel, resolving the ABI clash.
2.  **Conda Dynamic Linker / CXXABI Mismatch**:
    *   *Issue*: At startup, the compiler package `llvmlite` (used by `numba`/`quaternion`/`habitat-sim`) attempts to load the system `libstdc++.so.6` which is missing the `CXXABI_1.3.15` symbol.
    *   *Solution*: Prefix python commands with `LD_LIBRARY_PATH=/home/dani/miniconda3/envs/fine-eqa/lib`. This redirects the dynamic linker to load the more modern `libstdc++.so.6` bundled with the conda environment itself.

### 3. Verification Results
We verified the environment using the scratch script `verify_env.py`:
```bash
LD_LIBRARY_PATH=/home/dani/miniconda3/envs/fine-eqa/lib /home/dani/miniconda3/envs/fine-eqa/bin/python verify_env.py
```
**Output**:
*   **Habitat-Sim**: Loaded and initialized a default simulator config successfully.
*   **Prismatic VLM**: Imported successfully.
*   **PyTorch**: Working on GPU (Device: `NVIDIA GeForce RTX 5090`, Capability `12.0`). GPU tensor multiplication completes successfully.

### 4. Running the Code
To run the project, activate the environment and set the `LD_LIBRARY_PATH` before running the entry point:
```bash
conda activate fine-eqa
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
python main.py -cf fine_eqa.yaml
```
*(Note: Running `main.py` will initialize correctly, but will eventually halt when it tries to load dataset files or OpenAI keys if they are not configured in `fine_eqa.yaml` / `gpt.py` respectively.)*
