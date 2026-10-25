# Differential Gaussian Rasterization

Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>

## ParetoGS package

Install from the project root:

```bash
python -m pip install ./submodules/diff-gaussian-rasterization --no-build-isolation --no-deps
```

The Python package is `diff_gaussian_rasterization_paretogs`, with its own `_C`
extension. The required vector and matrix operations are implemented in
`cuda_rasterizer/raster_math.h`; GLM is not required.

This release also corrects the CUB Tile maximum reduction to match the actual
16×16 CUDA thread block. Earlier ParetoGS code used a 256×1 reduction, which can
produce invalid Tile bounds and skip valid backward contributions. This correction
can change training gradients and trajectories relative to the submission code.
Existing model parameters and the rendering equations are unchanged.

The build uses the local `setup.py`; the required headers and CUDA sources are
included in the source distribution. See the [main README](../../README.md) for
installation and the config-driven reproduction entry.
