## Dataset

Assume the ImageNet is in `/path/to/imagenet`. It should be like this:</summary>
    
```
/path/to/imagenet/:
    train/:
        n01440764: 
            many_images.JPEG ...
        n01443537:
            many_images.JPEG ...
    val/:
        n01440764:
            ILSVRC2012_val_00000293.JPEG ...
        n01443537:
            ILSVRC2012_val_00000236.JPEG ...
```

## Installation  
Install **pytorch_wavelets** for the PyTorch implementation of 2D discrete wavelet transforms.  
You can find it here: [pytorch_wavelets](https://github.com/fbcotter/pytorch_wavelets).

## Training Scripts

Training:

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 train_wav.py --data_path=/path/to/imagenet --bs 128 --wandb_name stage_1 --disc_loss_fn cross_entropy --rec_loss_fn focal
```

Evaluating:
```shell
CUDA_VISIBLE_DEVICES=0 python3 evaluator_vae_wav.py --data_path=/path/to/imagenet --bs 200 --load_ckpt_path=/path/to/checkpoint --rec_filename=/path/to/output/recon.png
```

### Parameters

- **`--ep`** *(default: 150)*  
  Number of training epochs.

- **`--bs`** *(default: 64)*  
  Batch size for training.

- **`--vae_blr`** *(default: 1e-4)*  
  Base learning rate for the VAE.

- **`--lc`** *(default: 1.0)*  
  Weight of the commitment loss.

- **`--lp`** *(default: 0.5)*  
  Weight of the LPIPS loss.

- **`--ld`** *(default: 1.0)*  
  Weight of the Discriminator loss.

- **`--rec_loss_fn`** *(default: l1)*
  Reconstruction loss function. Supported options: `l1`, `l2`, and `focal`.

- **`--disc_loss_fn`** *(default: hinge)*
  Discriminator loss function. Supported options: `hinge` and `cross_entropy`.

- **`--disc_blr`** *(default: 1e-4)*  
  Base learning rate for the Discriminator.

- **`--disc_start_ep`**
  Epoch at which Discriminator training begins. Training starts at `0.2 * ep` by default.

- **`--pn`** *(default: '1_2_3_4_5_6_8_10_13_16')*  
  Multi-scale patch size.

- **`--ch`** *(default: '1_2_4')*  
  Autoencoder channel multiplication

## Relevant Files

- **`train_wav.py`**  
  Script for training the VQVAE.

- **`trainer_wav.py`**  
  PyTorch Lightning trainer module.

- **`evaluator_vae_wav.py`**  
  Calculates the reconstruction loss of VQVAE on wavelet data.

- **`models/__init__.py`**  
  Builds and initializes models.

- **`models/basic_vae.py`**  
  Basic VAE model (no description provided).

- **`models/vqvae_wav.py`**  
  VQVAE implementation for wavelet data.

- **`models/loss.py`**
  Defines the loss functions for training.

- **`models/discriminator.py`**  
  Discriminator module, primarily based on [VQGAN](https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py).

- **`models/lpips.py`**  
  Implements VGG16 for calculating [LPIPS](https://github.com/richzhang/PerceptualSimilarity/tree/master/models).

- **`models/quant.py`**  
  Quantization module (no description provided).

- **`utils/arg_util.py`**  
  Handles argument parsing and configuration.

- **`utils/data_wav.py`**  
  PyTorch Lightning DataModule for wavelet data.