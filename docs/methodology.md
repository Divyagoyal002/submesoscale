# Methodology

## 1. Physical basis

At horizontal scales much larger than about 1 km and away from the equator, surface currents are in
**geostrophic balance**. The Coriolis force balances the pressure gradient set by the sea-surface slope:

$$u_g = -\frac{g}{f}\frac{\partial \eta}{\partial y}, \qquad v_g = \frac{g}{f}\frac{\partial \eta}{\partial x}, \qquad f = 2\Omega\sin\varphi$$

Here η is the Absolute Dynamic Topography (ADT). On the lat/lon grid, $dx = R\cos\varphi\,d\lambda$ and
$dy = R\,d\varphi$ (see `physics.py`). The **Rossby number** $Ro = \zeta/f$, where
$\zeta = \partial_x v - \partial_y u$, measures how far the flow departs from pure geostrophy. When
$|Ro| \sim O(1)$, the flow is submesoscale.

Currents depend on the *gradient* of η and vorticity on its *second derivative*. Small errors in η at short
wavelengths therefore become large errors in currents. That's why the loss and the diagnostics focus on
currents and vorticity rather than η alone.

### Why SST helps

In surface quasi-geostrophic (SQG) dynamics, surface buoyancy and streamfunction are linked in Fourier
space by $\hat b = N|k|\hat\psi$. SST anomalies therefore trace the same eddies and fronts as SSH, with
more weight at small scales. L4 SST products resolve features at about 5 km, compared with about 100 km
for L4 ADT. The network learns this relationship from the model and uses it to sharpen ADT.

## 2. OSSE: training data with a known answer

The high-resolution reanalysis (1/24° Mediterranean, 1/40° Black Sea) serves as the "truth". To make
inputs that look like the real L4 products, the truth is degraded (`data/osse.py`):

1. **ADT**: a NaN-aware Gaussian filter (σ = 18 km). A Gaussian with σ has its half-power wavelength
   at about 5.3σ, so 18 km gives roughly 100 km, close to the effective resolution of DUACS in the
   Mediterranean. The field is then sampled onto the **1/16° grid of the real product**, white mapping
   noise (1 cm) is added, and it is **cubically** interpolated back to the target grid. Cubic, not
   bilinear, because bilinear interpolation leaves grid-cell artefacts in the vorticity.
2. **SST**: light smoothing (σ = 4 km) plus 0.05 K noise.

Real L4 data goes through the *same* regridding function at inference time, so the network sees the
same kind of input in training and in use.

The splits are **by year** (train, then val, then test, then reconstruct), so days that are close
together in time never land on both sides of a split.

## 3. Network

Both networks use the pre-upsampling design. The LR ADT is already on the target grid, so the network
predicts only a **residual correction**:

$$\hat\eta = \eta_{LR} + \sigma_{res}\,\mathrm{CNN}\big([\tilde\eta_{LR}, \widetilde{SST}, \text{mask}]\big)$$

- **Inputs** are normalised per patch: the patch mean is removed, because basin-mean sea level and
  seasonal SST carry no fine-scale information. Each input is then divided by a scale computed on the
  training split. Land is set to 0, and the land mask is passed as a third channel.
- **`resnet`** (default) is EDSR-style: residual blocks with no batch-norm, residual scaling of 0.1,
  GELU activations, and dilations cycling 1, 2, 4, 2, which gives a receptive field of about 60 grid
  cells (about 200 km). The last layer is initialised to zero, so before training the network
  reproduces the LR input exactly.
- **`srcnn`** is the 3-layer network of Dong et al. (2015), kept as a baseline for ablation.

## 4. Physics-informed loss (`models/losses.py`)

$$\mathcal L = w_{adt}\frac{\|\hat\eta-\eta\|^2}{\sigma_{res}^2} + w_{geo}\frac{\|\mathbf u_g(\hat\eta)-\mathbf u_g(\eta)\|^2}{U^2} + w_{vort}\|Ro(\hat\eta)-Ro(\eta)\|^2 + w_{spec}\,\overline{\big|\log E_{\hat\eta}(k)-\log E_\eta(k)\big|}$$

| Term | What it constrains | Why it matters |
|---|---|---|
| ADT | The height field itself | Keeps the large scales anchored |
| Geostrophic | ∇η, i.e. the currents | The quantity the project actually cares about |
| Rossby number | ∇²η / f | Dominated by the submesoscale; **new in this project** |
| Spectral | Radially averaged power spectrum | Penalises both over-smoothing and invented noise; **new in this project** |

The derivative terms are scored only where the whole finite-difference stencil lies in the ocean: the
mask is eroded by one pixel per derivative order.

**Suggested ablation for the paper:** train with `-o loss.w_geo=0 -o loss.w_vort=0 -o loss.w_spec=0`
(ADT only), then add the terms back one at a time and compare `metrics_osse.json`.

## 5. Inference

Full-domain fields are processed in tiles that match the training patch size and overlap by 25%
(`inference.py`). Each tile gets the same per-tile normalisation as in training, and the tiles are
blended with a Hann taper so no seams appear.

## 6. Validation

### OSSE test year (the truth is known)
- RMSE of ADT, u, v and speed; correlation of speed and of Rossby number.
- **Effective resolution** (Ballarotta et al. 2019): the wavelength at which the spectral score
  $1 - PSD(\hat\eta-\eta)/PSD(\eta)$ falls to 0.5.
- **Band-energy ratio** over 10–50 km: $\int E_{\hat\eta}/\int E_\eta$. A value near 1 means the
  fine-scale energy is correct, below 1 means the field is over-smoothed, and above 1 means the network
  has added energy that isn't there.

### Real data (the truth is unknown)
- **Drifters** (Global Drifter Program, 6-hourly QC). Only drogued observations are kept, because
  those follow the currents at about 15 m depth and are only weakly pushed by the wind. They are
  averaged over 24 h to suppress inertial and tidal motions, and then compared with the L4 and SR
  currents at the same points. Drifters measure the *total* current (geostrophic plus Ekman plus
  other ageostrophic motion), so some absolute error can never be removed. The meaningful result is
  the **relative improvement of SR over L4**.
- **Spectra** of L4 and SR fields over the largest land-free square, with the fitted slope over 10–100 km.
  For reference, QG theory predicts about k⁻⁵ and SQG about k⁻¹¹ᐟ³.

## 7. Known limitations

- The reanalysis is itself a model at about 4 km resolution. It under-represents the energy below
  about 15 km, and so does the network trained on it.
- The OSSE uses white mapping noise, whereas real DUACS errors are spatially correlated and depend on
  where the satellite tracks fell.
- Geostrophy ignores the ageostrophic submesoscale flow (|Ro| ≳ 1), where cyclostrophic corrections matter.
- The results on synthetic data are optimistic, because synthetic SST follows SQG exactly.

## References

- Ciani, D., Fanelli, C., Buongiorno Nardelli, B. (2025). Estimating ocean currents from the joint reconstruction of absolute dynamic topography and sea surface temperature through deep learning algorithms. *Ocean Science* 21, 199–216. https://doi.org/10.5194/os-21-199-2025
- Dong, C., Loy, C. C., He, K., Tang, X. (2015). Image super-resolution using deep convolutional networks. *IEEE TPAMI*.
- Lim, B. et al. (2017). Enhanced deep residual networks for single image super-resolution (EDSR). *CVPR Workshops*.
- Ballarotta, M. et al. (2019). On the resolutions of ocean altimetry maps. *Ocean Science* 15, 1091.
- Lapeyre, G., Klein, P. (2006). Dynamics of the upper oceanic layers in terms of surface quasigeostrophy theory. *J. Phys. Oceanogr.* 36, 165.
- Kundu, P. K. (1976). Ekman veering observed near the ocean bottom. *J. Phys. Oceanogr.* 6, 238 (complex vector correlation).
