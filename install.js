module.exports = {
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install -e .[inference]",
          "uv pip install gradio"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env"
        }
      }
    },
    // Install Q8-Kernels for FP8 support (NVIDIA Ada architecture and later)
    {
      when: "{{gpu === 'nvidia' && platform !== 'darwin'}}",
      method: "shell.run",
      params: {
        venv: "env",
        path: "LTX-Video-Q8-Kernels.git",
        message: [
          "uv pip install packaging wheel ninja setuptools",
          "uv pip install --no-build-isolation -e ."
        ]
      }
    },
    {
      method: "fs.link",
      params: {
        venv: "env"
      }
    },
    // Clean up duplicate dist-info files to prevent version conflicts after linking
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "python scripts/cleanup_dist_info.py"
      }
    }
  ]
}
