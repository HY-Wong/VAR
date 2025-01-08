import argparse
import os
import numpy as np
import torch
import pywt

from torchvision import transforms
from PIL import Image
from multiprocessing import Pool
from functools import partial


def decompose_multilevel_2d(image, wavelet, mode='periodization', level=2):
    """
    Perform 2D wavelet decomposition with the specified level
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


def normalize_01_into_pn1(x):  
    """
    Normalize x from [0, 1] to [-1, 1] by (x * 2 - 1)
    """
    return x.add(x).add_(-1)


def process_images_in_class(class_id, split, dataset_dir, output_dir, decomposition_level=2):
    """
    Process images in a specific class folder and save the results
    """
    # transform image to (256, 256)
    final_reso = 256
    mid_reso = round(1.125 * final_reso)

    transform = transforms.Compose([
        transforms.Resize(mid_reso, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize_01_into_pn1
    ])

    class_path = os.path.join(dataset_dir, split, class_id)
    output_path = os.path.join(output_dir, split, class_id)
    os.makedirs(output_path, exist_ok=True)

    print(f'Processing {class_path}')

    for img_name in os.listdir(class_path):
        if img_name.endswith('.JPEG') or img_name.endswith('.png'):
            img_path = os.path.join(class_path, img_name)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            img_np = np.array(img, dtype=np.float32)

            # perform wavelet decomposition
            subbands = decompose_multilevel_2d(img_np, wavelet='haar', level=decomposition_level)

            ll = np.stack([sb[decomposition_level-1][0] for sb in subbands], axis=0)
            data = {'ll': torch.tensor(ll)}
            for level in range(decomposition_level):
                lh = np.stack([sb[level][1][0] for sb in subbands], axis=0)
                hl = np.stack([sb[level][1][1] for sb in subbands], axis=0)
                hh = np.stack([sb[level][1][2] for sb in subbands], axis=0)
                hs = np.concatenate([lh, hl, hh], axis=0)
                data[f'l{level+1}_hs'] = torch.tensor(hs)

            img_id = img_name.split('.')[0]
            pt_path = os.path.join(output_path, f'{img_id}.pt')
            torch.save(data, pt_path)


def process_split(split, dataset_dir, output_dir, decomposition_level=2):
    """
    Process a specific data split (train/val) in parallel
    """
    split_path = os.path.join(dataset_dir, split)
    class_ids = [cid for cid in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, cid))]

    # use multiprocessing pool to parallelize the processing
    pool = Pool(processes=8) 
    worker = partial(
        process_images_in_class,
        split=split,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        decomposition_level=decomposition_level
    )
    pool.map(worker, class_ids)  # distribute `class_ids` among processes
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and store wavelet coefficients from image dataset.')
    parser.add_argument(
        '--dataset_dir', type=str,
        default='/path/to/imagenet', help='Path to the input image data directory.'
    )
    parser.add_argument(
        '--output_dir', type=str,
        default=None, help='Path to the output data directory.'
    )
    parser.add_argument(
        '--level', type=int, default=2, help='Wavelet decomposition level.'
    )

    # parse arguments
    args = parser.parse_args()

    if args.dataset_dir == '/path/to/imagenet':
        raise ValueError(f'{"*"*20}  please specify --dataset_dir=/path/to/imagenet  {"*"*20}')
    if args.output_dir is None:
        args.output_dir = args.dataset_dir + '-wavelet'
    os.makedirs(args.output_dir, exist_ok=True)

    for split in ['train', 'val']:
        process_split(split, args.dataset_dir, args.output_dir, decomposition_level=args.level)