# Third-party notices

The rectangular Morton decoder and the dynamic symmetric int8 quantization /
Metal 4 TensorOps scheduling design in `h3_shaders.metal` are adapted from
ccv's Metal FlashAttention `NAMatMulKernel` and `NAInt8MatMulKernel`,
distributed under the following BSD-3-Clause license:

Copyright (c) 2010, Liu Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
- Neither the name of the authors nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## Upstream project

`h3.c` began as [antirez/h3.c](https://github.com/antirez/h3.c) by Salvatore
Sanfilippo, released under the MIT license reproduced in `LICENSE`. This fork
keeps that license and adds the CUDA backend and the web UI under the same
terms.

## Web UI dependencies

The web UI does not vendor any third-party source. Its dependencies are
installed from their own registries at build time and each keeps its own
license:

| Component | Where | License |
| --- | --- | --- |
| FastAPI, Starlette | backend | MIT |
| Uvicorn | backend | BSD-3-Clause |
| Pydantic, pydantic-settings | backend | MIT |
| python-multipart | backend | Apache-2.0 |
| argon2-cffi (Argon2 password hashing) | backend | MIT |
| React, React DOM | frontend | MIT |
| Vite, @vitejs/plugin-react | frontend | MIT |
| TypeScript | frontend | Apache-2.0 |
| ESLint, typescript-eslint | dev tooling | MIT |
| Playwright | dev tooling | Apache-2.0 |
| nginx | container image | BSD-2-Clause |
| FFmpeg | runtime dependency | LGPL-2.1-or-later or GPL-2.0-or-later, depending on the build |

The MiniMax-H3 checkpoint is **not** part of this repository and is covered by
its own license from MiniMax. No model weights of any kind are distributed
here, including for the optional post-processing stage.
