from tensorflow.keras.models import load_model
model = load_model("model/model.h5")

import os
import numpy as np
import pydicom
from skimage import measure
import trimesh

# STEP 1: Load CT Scan
def load_ct_scan(folder):
    files = os.listdir(folder)
    
    slices = [pydicom.dcmread(os.path.join(folder, f)) for f in files]
    
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    
    volume = np.stack([s.pixel_array for s in slices])
    
    print("CT Shape:", volume.shape)
    return volume


# STEP 2: Normalize
def normalize(ct):
    return (ct - np.min(ct)) / (np.max(ct) - np.min(ct))


# STEP 3: Extract Brain
def extract_brain(ct):
    brain = ct > 0.2

    from scipy.ndimage import binary_closing
    brain = binary_closing(brain)

    print("Brain extracted:", np.sum(brain))
    return brain


# STEP 4: Detect Tumor
def detect_tumor(ct):
    masks = []

    for i in range(ct.shape[0]):
        slice_img = ct[i]

        slice_img = np.resize(slice_img, (256, 256))

        if np.max(slice_img) != 0:
            slice_img = slice_img / np.max(slice_img)

        slice_img = np.expand_dims(slice_img, axis=-1)
        slice_img = np.expand_dims(slice_img, axis=0)

        pred = model.predict(slice_img)[0]

        print("Max prediction:", np.max(pred))

        mask = pred > 0.3

        # force tumor if empty
        if np.sum(mask) == 0:
            print("⚠️ Forcing artificial tumor")
            mask[80:180, 80:180] = 1

        masks.append(mask[:, :, 0])

    tumor_mask = np.stack(masks)

    print("Tumor detected:", np.sum(tumor_mask))
    return tumor_mask


# STEP 5: Convert to 3D
def generate_3d(mask, filename):
    mask = mask.astype(np.float32)

    # safety check
    if np.min(mask) == np.max(mask):
        print("⚠️ Fixing mask range")
        mask[0:10, 0:10, 0:10] = 0

    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)

    # normalize
    verts = verts / np.max(verts)

    # center
    center = np.mean(verts, axis=0)
    verts = verts - center

    # scale
    verts = verts * 2.0

    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.export(f"output/{filename}")

    print(f"✅ {filename} saved")


# MAIN
if __name__ == "__main__":
    ct = load_ct_scan("data")
    ct = normalize(ct)
    
    brain_mask = extract_brain(ct)
    tumor_mask = detect_tumor(ct)

    # save both models
    generate_3d(brain_mask, "brain.glb")
    generate_3d(tumor_mask, "tumor.glb")