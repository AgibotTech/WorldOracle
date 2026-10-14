"""Process-local execution adapters. Metric functions and batch shapes are unchanged."""
from concurrent.futures import ThreadPoolExecutor
import importlib
import math
import os
from pathlib import Path
from types import SimpleNamespace


def discover_frames(root, view):
    ia = importlib.import_module('acwmbench.metrics.interaction_spatial_alignment')
    root = Path(root)
    for directory in (root/view, root, root/'video'/view, root/'video'):
        if not directory.is_dir():
            continue
        with os.scandir(directory) as entries:
            files = sorted(Path(e.path) for e in entries
                           if Path(e.name).suffix.lower() in ia.IMAGE_EXTENSIONS and e.is_file())
        if files:
            result = {}
            for ordinal, path in enumerate(files):
                index = ia._frame_number(path)
                index = ordinal if index is None else index
                if index in result:
                    raise ValueError(f'Duplicate frame index {index}: {path}')
                result[index] = path
            return result
    raise FileNotFoundError(f'No frames under {root}, view={view}')


class SampledMasks:
    def __init__(self, array, stride):
        if array.ndim != 4:
            raise ValueError('Expected production four-dimensional masks')
        self.shape, self.ndim, self.stride = array.shape, array.ndim, stride
        self.data = array[:, ::stride].copy()

    def __getitem__(self, index):
        if not isinstance(index, tuple) or len(index) != 2:
            raise ValueError(f'Unsupported mask indexing: {index}')
        mask, frame = index
        if isinstance(frame, slice):
            start, stop, step = frame.indices(self.shape[1])
            if step <= 0 or start % self.stride or step % self.stride:
                raise ValueError('Sampling would change: unsupported slice')
            return self.data[mask, start//self.stride:(stop+self.stride-1)//self.stride:step//self.stride]
        if frame < 0 or frame % self.stride:
            raise ValueError('Sampling would change: unsupported frame')
        return self.data[mask, frame//self.stride]


def install_window(frame_limit):
    """Reference window adapter: original algorithms, unchanged frame coordinates."""
    if frame_limit is None:
        return
    kro = importlib.import_module('acwmbench.metrics.keyarea_visual_quality')
    ia = importlib.import_module('acwmbench.metrics.interaction_spatial_alignment')
    mvc = importlib.import_module('acwmbench.metrics.multiview_consistency')
    from acwmbench.io_utils import list_frame_files
    original_load, original_discover = kro.load_frames_as_array, ia._discover_frames

    def limited_load(path, indices=None, preferred_view=None):
        use = range(min(frame_limit, len(list_frame_files(path, preferred_view=preferred_view)))) if indices is None else [i for i in indices if 0 <= i < frame_limit]
        return original_load(path, indices=use, preferred_view=preferred_view)

    def limited_discover(*args, **kwargs):
        return {i: p for i, p in original_discover(*args, **kwargs).items() if 0 <= i < frame_limit}

    kro.load_frames_as_array = limited_load
    ia._discover_frames = mvc._discover_frames = limited_discover


def install(config, decode_workers=8, frame_limit=None):
    """Called once per worker. clear_episode() bounds mask memory across submissions."""
    import numpy as np
    from PIL import Image
    from acwmbench.io_utils import list_frame_files
    kro = importlib.import_module('acwmbench.metrics.keyarea_visual_quality')
    ia = importlib.import_module('acwmbench.metrics.interaction_spatial_alignment')
    mvc = importlib.import_module('acwmbench.metrics.multiview_consistency')
    stride = math.gcd(int(config['keyarea_visual_quality'].get('downsample_rate', 1)),
                      int(config['interaction_spatial_alignment'].get('downsample_rate', 10)))
    if stride < 1:
        raise ValueError('Invalid stride')
    original_frames, original_masks = kro.load_frames_as_array, ia._load_masks
    masks, models, lpips_models = {}, {}, {}

    def decode(path):
        with Image.open(path) as im:
            return np.asarray(im.convert('RGB'))

    class Frames:
        def __init__(self, path, preferred_view):
            self.files = list_frame_files(path, preferred_view=preferred_view)
            if frame_limit is not None:
                self.files = self.files[:frame_limit]
        def __len__(self):
            return len(self.files)
        def __getitem__(self, index):
            if not isinstance(index, slice):
                raise ValueError('Unsupported frame access')
            with ThreadPoolExecutor(max_workers=decode_workers) as pool:
                return np.stack(list(pool.map(decode, self.files[index])))

    def load_frames(path, indices=None, preferred_view=None):
        if indices is not None:
            if frame_limit is not None:
                indices = [i for i in indices if 0 <= i < frame_limit]
            return original_frames(path, indices=indices, preferred_view=preferred_view)
        return Frames(path, preferred_view)

    def load_masks(path):
        key = str(Path(path).resolve())
        if key not in masks:
            masks[key] = SampledMasks(original_masks(Path(path)), stride)
        return masks[key]

    original_dino = ia.DinoExtractor
    def dino(**kwargs):
        key = tuple(sorted((k, str(v)) for k, v in kwargs.items()))
        if key not in models:
            models[key] = original_dino(**kwargs)
        return models[key]

    original_lpips = kro.ensure_lpips
    def lpips_factory(**kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in lpips_models:
            lpips_models[key] = original_lpips().LPIPS(**kwargs)
        return lpips_models[key]

    kro.load_frames_as_array = load_frames
    kro._load_npz_masks = ia._load_masks = mvc._load_masks = load_masks
    def window_discover(*args, **kwargs):
        frames = discover_frames(*args, **kwargs)
        return frames if frame_limit is None else {i:p for i,p in frames.items() if 0 <= i < frame_limit}
    ia._discover_frames = mvc._discover_frames = window_discover
    ia.DinoExtractor = mvc.DinoExtractor = dino
    kro.ensure_lpips = lambda: SimpleNamespace(LPIPS=lpips_factory)
    return masks.clear
