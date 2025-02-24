import os
import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from trainer import VQVAE_Trainer
from models import build_vae_var
from utils import arg_util
from utils.data import ImageDataModule


if __name__ == '__main__':
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    
    # distributed training parameters
    gpus = torch.cuda.device_count()
    num_nodes = args.num_nodes
    rank = int(os.getenv('NODE_RANK')) if os.getenv('NODE_RANK') is not None else 0
    print(f'[INFO] Number of GPUs available: {gpus}')
    print(f'[INFO] Number of nodes: {num_nodes}')
    
    pl.seed_everything(args.seed, workers=True)

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
    data_module = ImageDataModule(args)
    train_loader = data_module.train_dataloader()

    # inspect the first batch
    imgs, labels = next(iter(train_loader))
    print(f'[INFO] Batch images shape: {imgs.shape}')
    print(f'[INFO] Cumulative batch size: {args.bs}')
    print(f'[INFO] Final learning rate: {args.vae_lr}')

    # build the model
    vae, _ = build_vae_var(
        device=None, patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        V=4096, Cvae=32, ch=160, share_quant_resi=4, test_mode=False,   # hard-coded VQVAE hyperparameters
        num_classes=100, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    
    if resume:
        model = VQVAE_Trainer.load_from_checkpoint(
            args.load_ckpt_path, vae=vae, args=args, steps_per_epoch=len(train_loader) // gpus
        )
        print(f'[INFO] Loaded from {args.load_ckpt_path}')
    else:
        model = VQVAE_Trainer(vae=vae, args=args, steps_per_epoch=len(train_loader) // gpus)
    
    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_ckpt_dir, 
        filename='{epoch:03d}', 
        save_last=True, # save the latest
        save_top_k=1, monitor='val_vae_rec_img_loss',  mode='min', # save the best based on vae_rec_img_loss
        every_n_epochs=args.save_every_n_epochs
    )
    callbacks = [LearningRateMonitor(logging_interval='step'), checkpoint_callback]
    
    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator='gpu', num_nodes=num_nodes, devices=gpus, 
        precision='32', deterministic=True,
        callbacks=callbacks, logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=args.max_ep
    )

    # train the model
    trainer.fit(model, data_module, ckpt_path=args.load_ckpt_path)

    # test the model
    # trainer.validate(model, data_module)
    trainer.test(model, data_module)