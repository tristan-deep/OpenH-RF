import os
os.environ["KERAS_BACKEND"] = "torch"

from pathlib import Path
from zea import File
from zea.beamform.pixelgrid import cartesian_pixel_grid
import numpy as np


OUTPUT = Path(__file__).parent / "openpros_sample.hdf5"

PATIENT_ID = '1' # 4 patient-level anatomies
PROSTATE_ID = '2021-03-16' # 62 prostate-level anatomies

data = np.load(f'3_0{PATIENT_ID}_P_{PROSTATE_ID}_data.npy', mmap_mode='r')[..., np.newaxis]
# data.shape: (1140, 40, 1000, 161, 1)

label = np.load(f'3_0{PATIENT_ID}_P_{PROSTATE_ID}_sos.npy', mmap_mode='r')
# label.shape: (1140, 1, 401, 161)

flag_single_sample = True # if True, only use the first sample for testing; if False, use all samples

# Dimensions
n_frames = 1 if flag_single_sample else data.shape[0]
n_tx = 20 
n_tx_side = n_tx // 2 # 10 sources per side
n_ax = 1000 # 1000 timesteps
n_el = 322 
n_el_side = n_el // 2 # 161 receivers per side
n_ch = 1 # RF
dt = 1e-7
nz, nx, ny = 401, n_el_side, 0
dx = dz = 3.75e-4

# Actual nx, nz, dx used during forward modeling. 
# Upsampling applied to achieve sufficient ppw.
upsample_factor = 3 
nx_up = nx * upsample_factor
nz_up = nz * upsample_factor
dx_up = dx / upsample_factor
dz_up = dz / upsample_factor
# dt_up = dt / 10 

raw_data = np.zeros((n_frames, n_tx, n_ax, n_el, 1), dtype=np.float32)
raw_data[:n_frames, :n_tx_side, :, :n_el_side] = data[:n_frames, :n_tx_side] # source surface, receiver surface
raw_data[:n_frames, :n_tx_side, :, n_el_side:] = data[:n_frames, n_tx_side:2*n_tx_side] # source surface, receiver rectum
raw_data[:n_frames, n_tx_side:, :, :n_el_side] = data[:n_frames, 3*n_tx_side:4*n_tx_side] # source rectum, receiver surface
raw_data[:n_frames, n_tx_side:, :, n_el_side:] = data[:n_frames, 2*n_tx_side:3*n_tx_side] # source rectum, receiver rectum

sos_values = label[:n_frames, 0, :, :, np.newaxis]
sos_coordinates = cartesian_pixel_grid(
    xlims=[0, (nx-1)*dx],
    zlims=[0, (nz-1)*dz],
    grid_size_x=nx,
    grid_size_z=nz,
)

data = {
    "raw_data": raw_data,
    "sos_map": {
        "values": sos_values,
        "coordinates": sos_coordinates
    }
}


# during simulation, the sources are placed on the upsampled grid
sx = np.around(np.linspace(0, nx_up-1, num=n_tx_side)) * dx_up
sx = np.tile(sx, 2)
sy = np.zeros_like(sx)
sz = np.array([1]*n_tx_side + [nz_up-1]*n_tx_side) * dz_up # half surface, half rectum
source_positions = np.stack([sx, sy, sz], axis=-1).astype(np.float32)

def ricker(f, dt, nt, dx):
    nw = 2.2/f/dt
    nw = 2*np.floor(nw/2)+1
    nc = np.floor(nw/2)
    k = np.arange(nw)
    
    alpha = (nc-k)*f*dt*np.pi
    beta = alpha ** 2
    w0 = (1-beta*2)*np.exp(-beta)
    w = np.zeros(nt)
    w[:len(w0)] = w0
    return 1 / (dx**2) * w

dt_up = 1e-8
nt_up = 1e5

source_wavelet = ricker(1e6, dt_up, int(nt_up), dx_up).astype(np.float32)

scan = {
    "sampling_frequency": np.float32(1e7),
    "center_frequency": np.float32(1e6),

    # Real RF data, not IQ-demodulated data
    "demodulation_frequency": np.float32(0.0), 
    "sound_speed": np.float32(1500.0),

    # Recording starts at the transmit reference time
    "initial_times": np.zeros(n_tx, dtype=np.float32), 

    # Not physically applicable: transmissions come from external point sources
    "t0_delays": np.zeros((n_tx, n_el), dtype=np.float32),
    "tx_apodizations": np.zeros((n_tx, n_el), dtype=np.float32),

    # Compatibility placeholders
    "focus_distances": np.zeros(n_tx, dtype=np.float32),  # 0 = plane wave
    "transmit_origins": source_positions,
    "polar_angles": np.zeros(n_tx, dtype=np.float32),
    "waveforms_one_way": np.tile(source_wavelet, (n_tx, 1)),
}


# Transducer positions
# Note: Transducers are also placed on the upsampled grids.
probe_geometry = np.column_stack([
    np.tile(np.linspace(0, n_el_side-1, num=n_el_side) * dx, 2),
    np.zeros(n_el, dtype=np.float32),   
    np.array([1]*n_el_side + [nz_up-1]*n_el_side) * dz_up, 
]).astype(np.float32)


File.create(
    path=str(OUTPUT),
    data=data,
    scan=scan,
    probe={ 
        "name": 'simulated linear array',
        "type": "linear",
        "probe_geometry": probe_geometry 
    }, # name N/A
    metadata={
        "credit": '''Wang, H., Wu, Y., Feng, Y., Jin, P., Zhang, L., Feng, S., ... & Lin, Y. (2026, April). 
         Openpros: A large-scale dataset for limited view prostate ultrasound computed tomography. 
         In International Conference on Learning Representations (Vol. 2026, pp. 113741-113759).
         corresponding author: yzlin@unc.edu, The University of North Carolina at Chapel Hill''',
         "subject": { 
             "type": "simulation",
             "id": f"{PATIENT_ID}_{PROSTATE_ID}",
         },
         "annotations": {
             "anatomy": "prostate and surrounding male pelvic anatomy",
         }
    },
    description="OpenPros Sample",
    overwrite=True,
)

with File(str(OUTPUT)) as f:
    print(f"Saved: {OUTPUT}")
    print(f"  raw_data     : {f.data.raw_data.shape}")
    print(f"  sos_map      : {f.data.sos_map.values.shape}")
    print(f.load_parameters())
