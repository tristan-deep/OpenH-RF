# SPDX-License-Identifier: CC-BY-4.0
"""Reference reconstruction for the 3D ring-array USCT zea sub-dataset.

Loads one converted ``.hdf5`` (zea) acquisition and reconstructs a reflectivity
image directly from the raw RF channel data, using the PURE-calibrated ring
geometry, sampling rate, and time-zero **read back from the zea file**. This is
the sanity check requested by the OpenH-RF guide.

The beamformer is the project's fused-Triton round-trip DAS (``das_kernel.py``,
class ``ReflectivityDASTriton``): for every pixel it sums the round-trip
delay-and-sum over all 64 transmit x 256 receive pairs, with additive
transmission rejection and backscatter (cosine) apodization. It is wrapped as a
**custom registered ``zea.ops.Operation``** (``ring_das_reflectivity_3d``) so the
reconstruction is a ``zea.Pipeline`` saved to ``pipeline_3d.yaml``.

Usage:
    python reconstruct_3d.py                       # first file in ./data/3d
    python reconstruct_3d.py data/3d/phantom_xxx.hdf5
    python reconstruct_3d.py --save-pipeline       # (re)write pipeline_3d.yaml and exit

Run in the `ut` env (the one with `zea` + `triton` installed).
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import zea
from zea import File
from zea.ops import Operation
from zea.internal.registry import ops_registry

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False

HERE = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
#  Fused-Triton round-trip DAS  (ported from das_kernel.py)
# ===========================================================================

if _HAVE_TRITON:
    @triton.jit
    def _das_kernel(
        wave_ptr, elx_ptr, ely_ptr, txx_ptr, txy_ptr,
        pkx_ptr, pky_ptr, out_ptr, hit_ptr,
        Tx, N, T, NP, dt, t0, inv_c0, guard_s, BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        m = offs < NP
        pxx = tl.load(pkx_ptr + offs, mask=m, other=0.0)
        pyy = tl.load(pky_ptr + offs, mask=m, other=0.0)
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        hit = tl.zeros((BLOCK,), dtype=tl.float32)
        for tx in range(0, Tx):
            ex_t = tl.load(txx_ptr + tx)
            ey_t = tl.load(txy_ptr + tx)
            dxt = ex_t - pxx
            dyt = ey_t - pyy
            dist_t = tl.sqrt(dxt * dxt + dyt * dyt)
            tof_t = dist_t * inv_c0
            inv_lt = 1.0 / (dist_t + 1e-9)
            uxt = dxt * inv_lt
            uyt = dyt * inv_lt
            for rx in range(0, N):
                ex_r = tl.load(elx_ptr + rx)
                ey_r = tl.load(ely_ptr + rx)
                dxr = ex_r - pxx
                dyr = ey_r - pyy
                dist_r = tl.sqrt(dxr * dxr + dyr * dyr)
                tof_r = dist_r * inv_c0
                t_round = tof_t + tof_r
                ddx = ex_t - ex_r
                ddy = ey_t - ey_r
                t_direct = tl.sqrt(ddx * ddx + ddy * ddy) * inv_c0
                keep = t_round > (t_direct + guard_s)
                fidx = (t_round - t0) / dt
                idx = (fidx + 0.5).to(tl.int32)
                valid = (idx >= 0) & (idx < T) & keep
                idx_c = tl.where(valid, idx, 0)
                base = (tx * N + rx) * T
                amp = tl.load(wave_ptr + base + idx_c, mask=valid, other=0.0).to(tl.float32)
                inv_lr = 1.0 / (dist_r + 1e-9)
                uxr = dxr * inv_lr
                uyr = dyr * inv_lr
                cosang = uxt * uxr + uyt * uyr
                cosang = tl.where(cosang > 0.0, cosang, 0.0)
                wgt = tl.where(valid & (cosang > 0.0), 1.0, 0.0)
                acc += amp * cosang * wgt
                hit += wgt
        out = acc / (hit + 1e-6)
        tl.store(out_ptr + offs, out, mask=m)
        tl.store(hit_ptr + offs, hit, mask=m)


class ReflectivityDASTriton:
    """Fused-kernel reflectivity DAS (apod_power fixed at 1)."""

    def __init__(self, el_pos, tx_pos, num_pixels=512, fov_size=0.13, c0=1500.0,
                 device="cuda", ring_mask_frac=1.0, pulse_len_s=3.333e-6,
                 guard_pulses=0.75, block=256):
        if not _HAVE_TRITON:
            raise RuntimeError("Triton is required for the 3D ring DAS "
                               "(64-element transmit subset).")
        self.num_pixels = num_pixels
        self.c0 = c0
        self.device = device
        self.pulse_len_s = float(pulse_len_s)
        self.guard_pulses = float(guard_pulses)
        self.block = block

        el = torch.tensor(el_pos, dtype=torch.float32)
        self.N = el.shape[0]
        self.elx = el[:, 0].contiguous().to(device)
        self.ely = el[:, 1].contiguous().to(device)
        txp = torch.tensor(tx_pos, dtype=torch.float32)
        self.Ntx = txp.shape[0]
        self.txx = txp[:, 0].contiguous().to(device)
        self.txy = txp[:, 1].contiguous().to(device)

        grid_1d = torch.linspace(-fov_size / 2, fov_size / 2, num_pixels)
        py, px = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
        px = px.flatten(); py = py.flatten()
        self.grid_shape = (num_pixels, num_pixels)
        self.NP_full = px.numel()
        r = torch.sqrt(px ** 2 + py ** 2)
        keep = r <= ring_mask_frac * (fov_size / 2)
        self.keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1).to(device)
        self.NP = int(self.keep_idx.numel())
        self.pkx = px[keep.cpu()].contiguous().to(device)
        self.pky = py[keep.cpu()].contiguous().to(device)

    @torch.no_grad()
    def __call__(self, waveform, dt, t0):
        """waveform [Tx, T, Rx] -> [H, H] reflectivity (fp32 on device)."""
        Tx, T, Rx = waveform.shape
        dev = self.device
        wave = waveform.permute(0, 2, 1).contiguous().to(dev).half()  # [Tx, N, T]
        out = torch.empty(self.NP, device=dev, dtype=torch.float32)
        hit = torch.empty(self.NP, device=dev, dtype=torch.float32)
        guard_s = self.guard_pulses * self.pulse_len_s
        grid = (triton.cdiv(self.NP, self.block),)
        _das_kernel[grid](
            wave, self.elx, self.ely, self.txx, self.txy,
            self.pkx, self.pky, out, hit,
            Tx, self.N, T, self.NP,
            float(dt), float(t0), float(1.0 / self.c0), float(guard_s),
            BLOCK=self.block,
        )
        full = torch.zeros(self.NP_full, device=dev, dtype=torch.float32)
        full[self.keep_idx] = out
        return full.view(*self.grid_shape).contiguous()


def to_bmode(img2d, dr_db=40.0):
    a = img2d.abs(); a = a / (a.max() + 1e-12)
    return (20.0 * torch.log10(a + 1e-6)).clamp(min=-dr_db)


# ===========================================================================
#  Custom zea operation
# ===========================================================================

@ops_registry("ring_das_reflectivity_3d")
class RingDAS3D(Operation):
    """Fused-Triton round-trip DAS reflectivity for the 3D PURE ring."""

    def __init__(self, num_pixels=512, fov_size=0.13, guard_pulses=0.75,
                 pulse_len_s=3.333e-6, dr_db=40.0, **kwargs):
        kwargs.setdefault("jit_compile", False)
        kwargs.setdefault("jittable", False)
        kwargs.setdefault("with_batch_dim", False)
        super().__init__(**kwargs)
        self.num_pixels = num_pixels
        self.fov_size = fov_size
        self.guard_pulses = guard_pulses
        self.pulse_len_s = pulse_len_s
        self.dr_db = dr_db

    def call(self, **kwargs):
        wf = kwargs[self.key]                      # (Tx, Time, Rx) torch
        # fp16-safe pre-normalisation (the kernel casts the wave to half).
        peak = wf.abs().max()
        if peak > 0:
            wf = wf / peak * 1000.0
        das = ReflectivityDASTriton(
            el_pos=kwargs["el_pos"], tx_pos=kwargs["tx_pos"],
            num_pixels=self.num_pixels, fov_size=self.fov_size,
            c0=kwargs["c0"], device=str(wf.device),
            pulse_len_s=self.pulse_len_s, guard_pulses=self.guard_pulses,
        )
        img = das(wf, dt=kwargs["dt"], t0=kwargs["t0"])
        return {self.output_key: to_bmode(img, dr_db=self.dr_db)}


def build_pipeline(num_pixels=512, fov_size=0.13):
    return zea.Pipeline(operations=[
        RingDAS3D(num_pixels=num_pixels, fov_size=fov_size,
                  key="raw_data", output_key="reflectivity"),
    ])


# ===========================================================================
#  Load one zea acquisition
# ===========================================================================

def load_acquisition(path):
    with File(str(path), "r") as f:
        raw = np.asarray(f.data.raw_data[:])          # (1, Tx, n_ax, n_el, 1)
        s = f.scan
        fs = float(s.sampling_frequency)
        t0 = float(np.asarray(s.initial_times)[0])
        c0 = float(s.sound_speed)
        pg = np.asarray(f.probe.probe_geometry)       # (n_el, 3) = (x, y, z)
        tx_apod = np.asarray(s.tx_apodizations)       # (n_tx, n_el)
        sos = np.asarray(f.data.sos_map.values)[0]                # (H, W) m/s
        atten = np.asarray(f.data.attenuation_map.values)[0]      # (H, W) dB/m/Hz
        coords = np.asarray(f.data.sos_map.coordinates)           # (H, W, 3)
        dx = float(coords[0, 1, 0] - coords[0, 0, 0])

    waveform = raw[0, :, :, :, 0]                     # (Tx, Time, Rx)
    # The DAS uses (y, x) element coordinates (matches the validated build_geometry).
    el_pos = pg[:, [1, 0]].astype(np.float32)         # (n_el, 2) = (y, x)
    tx_elem = tx_apod.argmax(axis=1)                  # firing element per Tx
    tx_pos = el_pos[tx_elem]                          # (n_tx, 2)
    return dict(waveform=waveform, el_pos=el_pos, tx_pos=tx_pos, fs=fs, t0=t0,
                c0=c0, sos=sos, atten=atten, dx=dx)


def _centre_crop(img, dx, fov):
    """Crop a centred map to a square field of view of fov metres."""
    h = img.shape[0]
    half = int(round(fov / dx / 2))
    c = h // 2
    return img[c - half:c + half, c - half:c + half]


# ===========================================================================
#  Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path, default=None,
                    help="zea .hdf5 acquisition (default: first under ./data/3d)")
    ap.add_argument("--out", type=Path, default=HERE / "example_output_3d.png")
    ap.add_argument("--pipeline", type=Path, default=HERE / "pipeline_3d.yaml")
    ap.add_argument("--num-pixels", type=int, default=512)
    ap.add_argument("--fov", type=float, default=0.13)
    ap.add_argument("--save-pipeline", action="store_true")
    args = ap.parse_args()

    if args.save_pipeline:
        build_pipeline(args.num_pixels, args.fov).to_yaml(str(args.pipeline))
        print(f"Wrote {args.pipeline}")
        return

    if args.pipeline.exists():
        pipeline = zea.Pipeline.from_config(zea.Config.from_path(str(args.pipeline)))
    else:
        pipeline = build_pipeline(args.num_pixels, args.fov)

    path = args.file or next((HERE / "data" / "3d").rglob("phantom_*.hdf5"))
    print(f"Reconstructing: {path}  (device={DEVICE})")

    acq = load_acquisition(path)
    print(f"  Tx={acq['waveform'].shape[0]}  T={acq['waveform'].shape[1]}  "
          f"fs={acq['fs']/1e6:.3f} MHz  t0={acq['t0']*1e6:.3f} us  c0={acq['c0']:.0f} m/s")
    print(f"  recon {args.num_pixels}px / {args.fov*1e3:.0f} mm FOV")

    outputs = pipeline(
        raw_data=torch.from_numpy(acq["waveform"].astype(np.float32)).to(DEVICE),
        el_pos=acq["el_pos"], tx_pos=acq["tx_pos"],
        dt=1.0 / acq["fs"], t0=acq["t0"], c0=acq["c0"],
    )
    das = outputs["reflectivity"].cpu().numpy()

    # GT cropped to the DAS field of view for a like-for-like comparison.
    sos_c = _centre_crop(acq["sos"], acq["dx"], args.fov)
    atten_c = _centre_crop(acq["atten"], acq["dx"], args.fov)

    half_mm = args.fov * 1e3 / 2
    ext = [-half_mm, half_mm, -half_mm, half_mm]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (das, "DAS reflectivity [dB]", "gray"),
        (sos_c, "GT sound speed [m/s]", "viridis"),
        (atten_c, "GT attenuation [dB/cm/MHz]", "magma"),
    ]
    for ax, (img, title, cmap) in zip(axes, panels):
        im = ax.imshow(img, cmap=cmap, extent=ext, origin="lower")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f"{path.name} — reference 3D DAS reconstruction from zea channel data",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
