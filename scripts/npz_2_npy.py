import numpy as np

npz_path = "/path/to/masks.npz"
out_path = "/path/to/masks_stacked.npy"

data = np.load(npz_path)

# 按 frame 编号排序
keys = sorted(data.keys(), key=lambda x: int(x.split("_")[-1]))

# 堆叠成 (n, 400, 640)
masks = np.stack([data[k] for k in keys], axis=0).astype(np.uint8)

print(f"Stacked shape: {masks.shape}")
np.save(out_path, masks)
print(f"Saved to {out_path}")
