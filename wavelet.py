import os
import numpy as np
import h5py
import pywt
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def decompose_multilevel_2d(image, wavelet, mode='periodization', level=2):
    """
    Perform 2D wavelet decomposition with the specified level.
    """
    channels = []
    # for each color channel
    for c in range(image.shape[0]):
        lowpass = image[c, ...]
        subbands = []
        for _ in range(level):
            coeffs = pywt.dwt2(lowpass, wavelet=wavelet, mode=mode)
            lowpass, highpass = coeffs[0], coeffs[1]  # approximation (LL) and detail coefficients (LH, HL, HH)
            subbands.append((lowpass, highpass))
        channels.append(subbands)
    return channels


def visualize_wavelet_components(ll, lh, hl, hh, level):
    components = [ll, lh, hl, hh]
    titles = ['LL', 'LH', 'HL', 'HH']
    
    plt.figure(figsize=(10, 10))
    for i, (comp, title) in enumerate(zip(components, titles)):
        comp = np.transpose(comp, (1, 2, 0))
        plt.subplot(2, 2, i + 1)
        plt.imshow(comp / comp.max())  # normalize for better visualization
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'coeffs-level-{level}.png')


dataset_dir = '../datasets/imagenet-100'
output_dir = '../datasets/imagenet-100-wavelet'
os.makedirs(output_dir, exist_ok=True)

# transform image to (256, 256)
final_reso = 256
mid_reso = round(1.125 * final_reso)

transform = transforms.Compose([
    transforms.Resize(mid_reso, interpolation=transforms.InterpolationMode.LANCZOS),
    transforms.CenterCrop((final_reso, final_reso)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

for split in ['train' , 'val']:
    split_path = os.path.join(dataset_dir, split)
    output_path = os.path.join(output_dir)
    os.makedirs(output_path, exist_ok=True)

    hdf5_file = os.path.join(output_path, f'{split}.h5')
    with h5py.File(hdf5_file, 'w') as h5f: 
        for class_id in os.listdir(split_path):
            class_path = os.path.join(split_path, class_id)
            if not os.path.isdir(class_path):
                continue
            
            # process each image in the class folder
            print(f'Processing {class_path}')

            for img_name in os.listdir(class_path):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(class_path, img_name)
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    img_np = np.array(img, dtype=np.float32)
                    # perform wavelet decomposition
                    decomposition_level = 2
                    subbands = decompose_multilevel_2d(img_np, wavelet='haar', level=decomposition_level)
                    
                    grp = h5f.create_group(f'{class_id}/{img_name}')

                    for level in range(decomposition_level):
                        ll = np.stack([sb[level][0] for sb in subbands], axis=0)
                        lh = np.stack([sb[level][1][0] for sb in subbands], axis=0)
                        hl = np.stack([sb[level][1][1] for sb in subbands], axis=0)
                        hh = np.stack([sb[level][1][2] for sb in subbands], axis=0)
                        coeffs = np.concatenate([ll, lh, hl, hh], axis=0)
                        # print(ll.shape)
                        # print(lh.shape)
                        # print(hl.shape)
                        # print(hh.shape)
                        # print(coeffs.shape)
                        # visualize_wavelet_components(ll, lh, hl, hh, level+1)

                        grp.create_dataset(f'level-{level+1}', data=coeffs, compression='gzip') # (LL, LH, HL, HH)