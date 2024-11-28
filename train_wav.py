import os
import torch
import pytorch_lightning as pl
import wandb

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger

from trainer_wav import VQVAE_WAV_Trainer
from models import build_vae
from utils import arg_util
from utils.data_wav import WaveletDataModule


if __name__ == '__main__':
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    # distributed training parameters
    gpus = torch.cuda.device_count()
    num_nodes = args.num_nodes
    rank = int(os.getenv('NODE_RANK')) if os.getenv('NODE_RANK') is not None else 0
    print(f'[INFO] Number of GPUs available: {gpus}')
    print(f'[INFO] Number of nodes: {num_nodes}')
    
    seed = args.seed if args.seed else 0
    pl.seed_everything(seed, workers=True)

    # logging, checkpointing and resuming
    save_ckpt_dir = f'{args.save_ckpt_dir}/{args.wandb_name}'
    resume = args.load_ckpt_path is not None

    if rank == 0:  # prevents from logging multiple times
        logger = WandbLogger(
            project=args.wandb_project, name=args.wandb_name, id=args.wandb_id,
            offline=False, resume='must' if resume else None
        )
    else:
        logger = WandbLogger(
            project=args.wandb_project, name=args.wandb_name,
            offline=True
        )
    
    # data loading
    data_module = WaveletDataModule(args)
    train_loader = data_module.train_dataloader()

    # inspect the first batch
    (l1_hs, l2_hs, ll), label = next(iter(train_loader))
    print(f'[INFO] Batch level 1 highs shape: {l1_hs.shape}')
    print(f'[INFO] Batch level 2 highs shape: {l2_hs.shape}')
    print(f'[INFO] Batch lows shape: {ll.shape}')
    print(f'[INFO] Batch label shape: {label.shape}')
    print(f'[INFO] Cumulative batch size: {args.bs}')
    print(f'[INFO] Final learning rate: {args.tlr}')

    # build the model
    vae = build_vae(
        patch_nums=(1, 2, 3, 4, 5, 6, 7, 8),
        V=4096, Cvae=128, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        init_vae=args.init_vae, init_vocab=args.init_vocab,
        ch_mult=(1, 2, 2, 2), in_channels=21
    )
    if resume:
        model = VQVAE_WAV_Trainer.load_from_checkpoint(args.load_ckpt_path, vae=vae, args=args)
        print(f'[INFO] Loaded from {args.load_ckpt_path}')
    else:
        model = VQVAE_WAV_Trainer(vae=vae, args=args)
    
    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_ckpt_dir, 
        filename='{epoch:03d}', 
        save_last=True, # save the latest
        save_top_k=1, monitor='val_loss',  mode='min', # save the best based on val_loss
        every_n_epochs=args.save_every_n_epochs
    )
    callbacks = [LearningRateMonitor(), checkpoint_callback]

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator='gpu', num_nodes=num_nodes, devices=gpus, 
        precision='32', deterministic=True,
        callbacks=callbacks, logger=logger, 
        max_epochs=args.ep
    )

    # train the model
    trainer.fit(model, data_module, ckpt_path=args.load_ckpt_path)

    # test the model
    trainer.test(model, data_module)

    # ensure wandb has stopped logging
    wandb.finish()
