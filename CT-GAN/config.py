import os
from os.path import join
import numpy as np
from keras import backend as K
from yacs.config import CfgNode as CN


_C = CN(
    new_allowed=True,
    init_dict=dict(
        # DATA LOCATION
        healthy_scans_raw="data/healthy_scans",  # path to directory where the healthy scans are. Filename is patient ID.
        healthy_coords="data/healthy_coords.csv",  # path to csv where each row indicates where a healthy sample is (format: filename, x, y, z). 'fileneame' is the folder containing the dcm files of that scan or the mhd file name, slice is the z axis
        healthy_samples="data/healthy_samples.npy",  # path to pickle dump of processed healthy samples for training.
        unhealthy_scans_raw="data/unhealthy_scans/",  # path to directory where the unhealthy scans are
        unhealthy_coords="data/unhealthy_coords.csv",  # path to csv where each row indicates where a healthy sample is (format: filename, x, y ,z)
        unhealthy_samples="data/unhealthy_samples.npy",  # path to pickle dump of processed healthy samples for training.
        traindata_coordSystem="world",  # the coord system used to note the locations of the evidence ('world' or 'vox'). vox is array index.
        # MODEL LOCATION
        modelpath_inject=join(
            "data", "models", "INJ"
        ),  # path to save/load trained models and normalization parameters for injector
        modelpath_remove=join(
            "data", "models", "REM"
        ),  # path to save/load trained models and normalization parameters for remover
        progress="images",  # path to save snapshots of training progress
        # TENSORFLOW CONFIG
        devices=K.tensorflow_backend._get_available_gpus(),
        gpus="0"
        if len(devices) > 0
        else "",  # sets which GPU to use (use_CPU:"", use_GPU0:"0", etc...)
        # CT-GAN Configuration
        cube_shape=np.array([32, 32, 32]),  # z,y,x
        mask_xlims=np.array([6, 26]),
        mask_ylims=np.array([6, 26]),
        mask_zlims=np.array([6, 26]),
        copynoise=True,  # If true, the noise touch-up is copied onto the tampered region from a hardcoded coordinate. If false, gaussain interpolated noise is added instead
    ),
)

cfg = _C

assert (
    cfg.mask_zlims[1] > cfg.cube_shape[0]
), "Out of bounds: cube mask is larger then cube on dimension z."
assert (
    cfg.mask_ylims[1] > cfg.cube_shape[1]
), "Out of bounds: cube mask is larger then cube on dimension y."
assert (
    cfg.mask_xlims[1] > cfg.cube_shape[2]
), "Out of bounds: cube mask is larger then cube on dimension x."

# Make save directories
if not os.path.exists(cfg.modelpath_inject):
    os.makedirs(cfg.modelpath_inject)
if not os.path.exists(cfg.modelpath_remove):
    os.makedirs(cfg.modelpath_remove)
if not os.path.exists(cfg.progress):
    os.makedirs(cfg.progress)
