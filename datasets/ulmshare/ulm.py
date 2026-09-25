# SPDX-License-Identifier: Apache-2.0
"""Standard ULM localization + tracking on a clutter-filtered IQ movie.

A deliberately small, pure-Python (NumPy/SciPy) stand-in for the MATLAB TAL
toolbox (``TrackingAndLocalizationULM``) that ``example_script_one_buffer.m``
uses for its ULM stage. It is *not* a port of TAL — it is the textbook ULM
pipeline (Errico et al. 2015 / PALA, Heiles et al. 2022):

1. **Localize** — in every clutter-filtered frame, detect microbubbles as local
   maxima of the envelope above a noise floor, then refine each to sub-pixel
   position with a 3-point parabolic (log-Gaussian) interpolation on the
   magnitude. See :func:`localize_frame`.
2. **Track** — link localizations frame-to-frame with a nearest-neighbour
   assignment (Hungarian, :func:`scipy.optimize.linear_sum_assignment`) capped
   at ``max_linking_distance`` pixels, and keep only tracks lasting at least
   ``min_track_length`` frames. See :func:`track`.
3. **Interpolate** — smooth and temporally up-sample each track with a smoothing
   spline, the analogue of TAL's ``interpolation.Interpolator``. See
   :func:`interpolate_track`.
4. **Accumulate** — splat every interpolated position onto a super-resolution
   grid (``super_res`` × finer than the beamforming grid) to form the ULM
   density map. See :func:`density_map`.

Steps 3-4 mirror ``example_script_one_buffer.m``'s ``interp_tracks`` ->
``lib.DensityMappingND``. Step 3 matters more to the final image than the grid
pitch does: without it the map is a scatter of one dot per bubble per frame,
and making the grid finer only spreads those dots further apart.

Everything works in **beamforming-grid pixel coordinates** (the axes of the
``iq_cf`` movie from ``reconstruct.py``); the density map is returned on the
up-sampled grid. Distances/thresholds are therefore in pixels, which keeps the
code and its parameters simple and grid-relative.

Coordinate convention: a localization is ``(z, x)`` in fractional pixel units of
the ``(Nz, Nx)`` frame, matching NumPy row/column order.
"""

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

# Temporal up-sampling factor for track interpolation, matching TAL's
# ``config.json`` ``optInterpolation.interpFactor``.
INTERP_FACTOR = 10


def localize_frame(frame, threshold, min_distance=1):
    """Localize microbubbles in one clutter-filtered frame.

    Finds local maxima of ``|frame|`` that (a) exceed ``threshold`` and (b) are
    the largest value in their ``(2*min_distance+1)`` neighbourhood, then refines
    each maximum to sub-pixel accuracy with a 3-point parabolic fit of
    ``log|frame|`` along each axis (equivalent to fitting a Gaussian peak, the
    standard ULM sub-pixel estimator).

    Args:
        frame (np.ndarray): Complex (or real) frame ``(Nz, Nx)``.
        threshold (float): Magnitude floor; maxima at or below it are ignored.
        min_distance (int): Half-width of the suppression neighbourhood, in
            pixels. ``1`` means 3x3 (reject non-strict local maxima).

    Returns:
        np.ndarray: ``(n_bubbles, 2)`` array of ``(z, x)`` sub-pixel positions.
            Empty ``(0, 2)`` array if nothing is detected.
    """
    mag = np.abs(frame)
    size = 2 * min_distance + 1
    is_peak = (mag == maximum_filter(mag, size=size)) & (mag > threshold)
    # Drop peaks on the border: the parabolic fit needs both neighbours.
    is_peak[0, :] = is_peak[-1, :] = is_peak[:, 0] = is_peak[:, -1] = False

    zs, xs = np.nonzero(is_peak)
    if zs.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    # 3-point parabolic interpolation of log-magnitude for sub-pixel offset:
    # for samples a (left/up), b (centre), c (right/down),
    #   offset = 0.5 * (a - c) / (a - 2b + c), clamped to [-0.5, 0.5].
    logmag = np.log(mag + 1e-12)

    dz = _sub_offset(logmag[zs - 1, xs], logmag[zs, xs], logmag[zs + 1, xs])
    dx = _sub_offset(logmag[zs, xs - 1], logmag[zs, xs], logmag[zs, xs + 1])

    return np.stack([zs + dz, xs + dx], axis=1)


def _sub_offset(up, mid, down):
    """3-point parabolic peak offset from samples ``up``/``mid``/``down``, clamped to +-0.5."""
    denom = up - 2.0 * mid + down
    off = np.where(np.abs(denom) > 1e-12, 0.5 * (up - down) / denom, 0.0)
    return np.clip(off, -0.5, 0.5)


def localize_movie(iq_cf, threshold_snr=2.0, min_distance=1):
    """Localize microbubbles in every frame of a clutter-filtered movie.

    The magnitude threshold is set from the movie's noise level: ``threshold_snr``
    times the median magnitude over the whole movie (a robust, speckle-insensitive
    proxy for the noise floor).

    A per-depth-row noise floor is a tempting refinement, since the
    clutter-filtered floor does vary ~4x across the 8 mm range — but it was tried
    and measurably *hurt* agreement with the MATLAB reference, so it is
    deliberately not offered. Near the surface the per-row floor is about half the
    global one, dropping the threshold far enough that the tracker links speckle
    into a stippled arc of false vessels across the top of the image.

    Args:
        iq_cf (np.ndarray): Clutter-filtered complex movie ``(n_frames, Nz, Nx)``.
        threshold_snr (float): Detection threshold as a multiple of the median
            magnitude.
        min_distance (int): Local-maximum suppression half-width, in pixels.

    Returns:
        list[np.ndarray]: Per-frame ``(n_bubbles, 2)`` arrays of ``(z, x)``
            sub-pixel positions.
    """
    mag = np.abs(iq_cf)
    threshold = threshold_snr * float(np.median(mag))
    return _localize_stack(mag, threshold, min_distance=min_distance)


def _localize_stack(mag, threshold, min_distance=1):
    """:func:`localize_frame` over a whole ``(n_frames, Nz, Nx)`` magnitude stack at once.

    One ``maximum_filter`` over the stack (size 1 along the frame axis, so frames
    stay independent) and one vectorised sub-pixel fit replace a Python loop over
    frames; the result is identical, frame by frame, to :func:`localize_frame`.
    """
    size = 2 * min_distance + 1
    is_peak = (mag == maximum_filter(mag, size=(1, size, size))) & (mag > threshold)
    is_peak[:, 0, :] = is_peak[:, -1, :] = is_peak[:, :, 0] = is_peak[:, :, -1] = False

    fs, zs, xs = np.nonzero(is_peak)  # sorted by frame
    # Log-magnitude only where the fit needs it, not over the whole stack.

    def logmag(dz, dx):
        return np.log(mag[fs, zs + dz, xs + dx] + 1e-12)

    mid = logmag(0, 0)
    dz = _sub_offset(logmag(-1, 0), mid, logmag(1, 0))
    dx = _sub_offset(logmag(0, -1), mid, logmag(0, 1))

    pts = np.stack([zs + dz, xs + dx], axis=1)
    bounds = np.searchsorted(fs, np.arange(1, mag.shape[0]))
    return np.split(pts, bounds)


def track(
    localizations,
    max_linking_distance=2.0,
    min_track_length=15,
    max_gap=2,
):
    """Link per-frame localizations into tracks and return their points.

    Frame-to-frame linking is a Hungarian (optimal) nearest-neighbour
    assignment: for each consecutive frame pair, points are matched to minimise
    total displacement. Two details matter for yield, and getting either wrong
    fragments tracks badly enough that almost nothing survives
    ``min_track_length``:

    * The ``max_linking_distance`` gate is applied **inside the cost matrix**
      (forbidden pairs cost infinity) rather than by filtering the solution
      afterwards. Filtering afterwards lets the optimizer spend a point on a
      too-far pairing that is then discarded, when a nearer candidate was still
      free — so a track dies while its true continuation goes unclaimed.
    * A track is not killed the first frame it goes unmatched. Microbubbles
      routinely dip below the detection threshold for a frame or two, so a track
      may coast unmatched for up to ``max_gap`` frames and still be revived.
      Coasted frames are recorded as NaN and filled in afterwards by linear
      interpolation, mirroring TAL's ``fillmissing(track, "makima", 1)``.

    Args:
        localizations (list[np.ndarray]): Per-frame ``(n, 2)`` ``(z, x)`` arrays
            from :func:`localize_movie`.
        max_linking_distance (float): Max frame-to-frame jump, in pixels. With
            ``max_gap``, a coasting track is allowed proportionally further:
            a bubble missed for ``g`` frames may have moved ``g+1`` steps.
        min_track_length (int): Minimum number of frames a track must span.
        max_gap (int): Frames a track may go unmatched before being finalized.
            ``0`` restores the strict frame-to-frame behaviour.

    Returns:
        list[np.ndarray]: One ``(track_len, 2)`` array of ``(z, x)`` positions
            per surviving track, gaps filled.
    """
    if not len(localizations):
        return []

    # Each active track: {"pts": [(z, x) or (nan, nan), ...], "miss": int}.
    active = [{"pts": [tuple(p)], "miss": 0} for p in localizations[0]]
    finished = []

    for frame_pts in localizations[1:]:
        matched_cur = set()
        matched_active = set()

        if active and len(frame_pts):
            # Predict from each track's last *known* position, and allow a
            # track that has been coasting to have travelled further.
            prev_pts = np.array([_last_known(t["pts"]) for t in active])
            gate = max_linking_distance * (1.0 + np.array([t["miss"] for t in active], dtype=float))

            cost = np.linalg.norm(prev_pts[:, None, :] - frame_pts[None, :, :], axis=2)
            # Gate *inside* the cost: forbidden pairings are never chosen, so
            # the optimizer only ever assigns links it is allowed to keep.
            cost = np.where(cost <= gate[:, None], cost, np.inf)

            # linear_sum_assignment rejects all-inf rows/cols, so solve on the
            # sub-problem of tracks and points that have at least one option.
            row_ok = np.isfinite(cost).any(axis=1)
            col_ok = np.isfinite(cost).any(axis=0)
            if row_ok.any() and col_ok.any():
                sub = cost[np.ix_(row_ok, col_ok)]
                # A finite-but-large fill keeps the sub-problem feasible; any
                # pairing that lands on one is rejected by the check below.
                big = np.isfinite(sub).max(initial=0.0) + 1.0
                rows, cols = linear_sum_assignment(np.where(np.isfinite(sub), sub, big))
                row_map = np.nonzero(row_ok)[0]
                col_map = np.nonzero(col_ok)[0]
                for r, c in zip(rows, cols):
                    ri, ci = row_map[r], col_map[c]
                    if np.isfinite(cost[ri, ci]):
                        active[ri]["pts"].append(tuple(frame_pts[ci]))
                        active[ri]["miss"] = 0
                        matched_active.add(ri)
                        matched_cur.add(ci)

        # Unmatched tracks coast (NaN placeholder) until they exceed max_gap.
        still_active = []
        for i, t in enumerate(active):
            if i in matched_active:
                still_active.append(t)
            elif t["miss"] < max_gap:
                t["pts"].append((np.nan, np.nan))
                t["miss"] += 1
                still_active.append(t)
            else:
                finished.append(t)
        active = still_active

        # Start fresh tracks for unclaimed localizations.
        for c, p in enumerate(frame_pts):
            if c not in matched_cur:
                active.append({"pts": [tuple(p)], "miss": 0})

    finished.extend(active)

    out = []
    for t in finished:
        pts = _trim_trailing_nan(np.array(t["pts"], dtype=np.float64))
        if len(pts) >= min_track_length:
            out.append(_fill_gaps(pts))
    return out


def _last_known(pts):
    """Last non-NaN ``(z, x)`` in a track's point list."""
    for p in reversed(pts):
        if not np.isnan(p[0]):
            return p
    return pts[-1]


def _trim_trailing_nan(pts):
    """Drop the coasted (NaN) tail a track accumulated before being finalized."""
    keep = np.nonzero(~np.isnan(pts[:, 0]))[0]
    return pts[: keep[-1] + 1] if len(keep) else pts[:0]


def _fill_gaps(pts):
    """Linearly fill NaN gaps left by coasting, as TAL's ``fillmissing`` does."""
    idx = np.arange(len(pts))
    known = ~np.isnan(pts[:, 0])
    if known.all():
        return pts
    out = pts.copy()
    for k in range(2):
        out[:, k] = np.interp(idx, idx[known], pts[known, k])
    return out


def interpolate_track(track_pts, interp_factor=INTERP_FACTOR):
    """Smooth and up-sample one track in time, as TAL's ``Interpolator`` does.

    The Python analogue of ``interpolation.Interpolator.spline_interpolation``,
    which ``example_script_one_buffer.m`` applies before density mapping. A track
    holds one localization per frame, so splatting it directly leaves a string of
    isolated dots; up-sampling in time makes each track deposit a near-continuous
    line, which is what gives a ULM density map its vessel-like appearance.

    Two details follow the MATLAB (TAL ``config.json``: ``interpFactor: 10``,
    ``splineFrequency: 45``):

    * The track is resampled on a **time** vector at ``interp_factor`` times the
      frame rate — not by arc length. Fast bubbles therefore leave more widely
      spaced samples than slow ones, exactly as in TAL.
    * The spline is a **smoothing** spline (MATLAB ``csaps``), not an
      interpolating one, so localization jitter is filtered out rather than
      faithfully traced. SciPy's ``make_smoothing_spline`` is the closest
      equivalent; its regularisation is parameterised differently from ``csaps``'
      ``p``, so this is a behavioural match, not a numerical one.

    Since the frame rate scales the time vector and the output only depends on
    the *ratio* of sample spacing to duration, frames are used directly as the
    time base (equivalent up to that constant).

    Args:
        track_pts (np.ndarray): ``(track_len, 2)`` array of ``(z, x)`` positions,
            one per frame.
        interp_factor (int): Temporal up-sampling factor.

    Returns:
        np.ndarray: ``(n_samples, 2)`` array of ``(z, x)`` positions along the
            track. Returns the input unchanged if it is too short to fit.
    """
    n = len(track_pts)
    if n < 2:
        return track_pts

    t = np.arange(n, dtype=np.float64)
    t_interp = np.linspace(0.0, n - 1, (n - 1) * interp_factor + 1)

    # make_smoothing_spline needs >= 5 points; CubicSpline needs >= 4.
    if n >= 5:
        from scipy.interpolate import make_smoothing_spline

        return np.stack(
            [make_smoothing_spline(t, track_pts[:, k])(t_interp) for k in range(2)],
            axis=1,
        )
    if n >= 4:
        from scipy.interpolate import CubicSpline

        return CubicSpline(t, track_pts, axis=0)(t_interp)

    return np.stack([np.interp(t_interp, t, track_pts[:, k]) for k in range(2)], axis=1)


def density_map(tracks, grid_shape, super_res=10):
    """Accumulate tracked positions into a super-resolution ULM density map.

    Each track is first smoothed and temporally up-sampled
    (:func:`interpolate_track`, the analogue of TAL's ``Interpolator.interp_tracks``)
    so it reads as a line rather than a dotted trail — the single biggest factor
    in how the map looks. Points are then splatted onto a grid ``super_res`` times
    finer than the beamforming grid, rounding to the nearest cell exactly as
    TAL's ``lib.DensityMappingND`` does; the pixel value is the accumulated
    number of track points in that cell.

    Args:
        tracks (list[np.ndarray]): ``(track_len, 2)`` ``(z, x)`` arrays, in
            beamforming-grid pixel units, from :func:`track`.
        grid_shape (tuple[int, int]): ``(Nz, Nx)`` of the beamforming grid.
        super_res (int): Up-sampling factor per axis.

    Returns:
        np.ndarray: Density map ``(Nz*super_res, Nx*super_res)`` (float counts).
    """
    nz, nx = grid_shape
    out_shape = (nz * super_res, nx * super_res)
    dens = np.zeros(out_shape, dtype=np.float64)
    if not tracks:
        return dens

    pts = np.concatenate([interpolate_track(t) for t in tracks], axis=0)
    zi = np.round(pts[:, 0] * super_res).astype(int)
    xi = np.round(pts[:, 1] * super_res).astype(int)
    valid = (zi >= 0) & (zi < out_shape[0]) & (xi >= 0) & (xi < out_shape[1])
    np.add.at(dens, (zi[valid], xi[valid]), 1.0)
    return dens


def reconstruct_ulm(
    iq_cf,
    threshold_snr=2.0,
    min_distance=1,
    max_linking_distance=2.0,
    min_track_length=15,
    max_gap=4,
    super_res=10,
):
    """Run the full ULM pipeline on a clutter-filtered movie.

    Convenience wrapper chaining :func:`localize_movie` -> :func:`track` ->
    :func:`density_map`.

    Args:
        iq_cf (np.ndarray): Clutter-filtered complex movie ``(n_frames, Nz, Nx)``.
        threshold_snr (float): Detection threshold (× median magnitude).
        min_distance (int): Local-maximum suppression half-width, in pixels.
        max_linking_distance (float): Max frame-to-frame jump, in pixels.
        min_track_length (int): Minimum frames a track must span.
        max_gap (int): Frames a track may go undetected and still continue.
        super_res (int): Density-map up-sampling factor per axis.

    Returns:
        tuple: ``(density, tracks)`` — the ``(Nz*super_res, Nx*super_res)``
            density map and the list of surviving tracks.
    """
    localizations = localize_movie(iq_cf, threshold_snr=threshold_snr, min_distance=min_distance)
    tracks = track(
        localizations,
        max_linking_distance=max_linking_distance,
        min_track_length=min_track_length,
        max_gap=max_gap,
    )
    print(
        f"  ULM: {sum(len(p) for p in localizations)} localizations -> "
        f"{len(tracks)} tracks (>= {min_track_length} frames, "
        f"{sum(len(t) for t in tracks)} points)"
    )
    return density_map(tracks, iq_cf.shape[1:], super_res=super_res), tracks
