from PIL import Image
import numpy as np

depth_path = "/path/to/depth/0000001.png"
depth = np.array(Image.open(depth_path))
