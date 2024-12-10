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
CUDA_VISIBLE_DEVICES=0,1 python3 train_wav.py --data_path=/path/to/wavelet --bs 384 --wandb_name=stage_1_vqvae_wav
```

### Parameters

- **`--ep`** *(default: 250)*  
  Number of training epochs.

- **`--bs`** *(default: 64)*  
  Batch size for training.

- **`--vae_blr`** *(default: 3e-4)*  
  Base learning rate for the VAE.

- **`--lc`** *(default: 10.0)*  
  Weight of the commitment loss.

- **`--lp`** *(default: 5.0)*  
  Weight of the LPIPS loss.

## Relevant Files

- **`train_wav.py`**  
  training VQ-VAE

- **`trainer_wav.py`**  
  PyTorch Lightning trainer

- **`wavelet.py`**  
  creating the wavelet dataset

- **`utils/arg_util.py`**  
  all argument parsing and configuration

- **`utils/data_wav.py`**  
  PyTorch Lightning DataModule

- **`models/__init__.py`**  
  building models and initialization

- **`models/basic_vae.py`**  

- **`models/lpips.py`**  
  VGG16 to calculate LPIPS loss, original GitHub repository: [LPIPS](https://github.com/richzhang/PerceptualSimilarity/tree/master/models)

- **`models/quant.py`**  

- **`models/vqvae_wav.py`**  
  VQ-VAE for wavelet data
