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

## Training Scripts

Creating wavelet dataset (takes some time to run):

```shell
python3 wavelet.py --dataset_dir=/path/to/imagenet
```

Training:

```shell
CUDA_VISIBLE_DEVICES=0,1 python3 train_wav.py --data_path=/path/to/wavelet --bs 256 --wandb_name=stage_1 --ch 1_2_4 --pn 1_2_3_4_5_6_8_10_13_16
```

### Parameters

- **`--ep`** *(default: 250)*  
  Number of training epochs.

- **`--bs`** *(default: 64)*  
  Batch size for training.

- **`--vae_blr`** *(default: 1e-4)*  
  Base learning rate for the VAE.

- **`--lc`** *(default: 1.0)*  
  Weight of the commitment loss.

- **`--lp`** *(default: 5.0)*  
  Weight of the LPIPS loss.

- **`--ld`** *(default: 1.0)*  
  Weight of the Discriminator loss.

- **`--pn`** *(default: '1_2_3_4_5_6_7_8')*  
  Multi-scale patch size.

- **`--ch`** *(default: '1_2_2_2')*  
  Autoencoder channel multiplication

## Relevant Files

- **`train_wav.py`**  
  Script for training the VQVAE.

- **`trainer_wav.py`**  
  PyTorch Lightning trainer module.

- **`wavelet.py`**  
  Module for creating the wavelet dataset.

- **`evaluator_vae.py`**  
  Calculates the reconstruction loss of VQVAE on image data.

- **`evaluator_vae_wav.py`**  
  Calculates the reconstruction loss of VQVAE on wavelet data.

- **`utils/arg_util.py`**  
  Handles argument parsing and configuration.

- **`utils/data_wav.py`**  
  PyTorch Lightning DataModule for wavelet data.

- **`models/__init__.py`**  
  Builds and initializes models.

- **`models/basic_vae.py`**  
  Basic VAE model (no description provided).

- **`models/discriminator.py`**  
  Discriminator module, primarily based on [VQGAN](https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py).

- **`models/lpips.py`**  
  Implements VGG16 for calculating [LPIPS](https://github.com/richzhang/PerceptualSimilarity/tree/master/models).

- **`models/quant.py`**  
  Quantization module (no description provided).

- **`models/vqvae_wav.py`**  
  VQVAE implementation for wavelet data.