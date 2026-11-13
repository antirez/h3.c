"""Implementation of tools/h3_image_reference.py, split by area.

common.py    constants, fixture I/O, shared model loaders
t2i.py       text-to-image stages: text, block0, step, denoise, vae, e2e, all
f32truth.py  float32 truth stages: f32truth, step_large, and the edit F32 helpers
edit.py      reference-image editing stages: edit_vae and the edit_* stages
cli.py       stage table and command line
"""
