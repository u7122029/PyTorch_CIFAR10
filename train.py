import os
from argparse import ArgumentParser

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

from data import CIFAR10Data
from module import CIFAR10Module


def main(args):

    if bool(args.download_weights):
        CIFAR10Data.download_weights()
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{args.classifier}",
        filename="best",
        monitor="val_acc",
        mode="max",
        save_last=False
    )
    trainer = Trainer(
        #fast_dev_run=bool(args.dev),
        #gpus=-1,
        #weights_summary=None,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint],
        precision=args.precision
    )

    data = CIFAR10Data(args)
    model = CIFAR10Module(vars(args))
    if bool(args.pretrained):
        state_dict = os.path.join(
            "cifar10_models", "state_dicts", args.classifier + ".pt"
        )
        model.model.load_state_dict(torch.load(state_dict))
        trainer.test(model, data.test_dataloader())
        return

    trainer.fit(model, data.train_dataloader(), data.val_dataloader())
    #trainer.test(model, data.test_dataloader()) # validation dataset is the same as the testing dataset. No need to test.

    if args.save_best:
        # Load best checkpoint
        loaded = CIFAR10Module.load_from_checkpoint(f"checkpoints/{args.classifier}/best.ckpt", hparams=vars(args))
        torch.save(loaded.model.state_dict(), f"cifar10_models/state_dicts/{args.classifier}.pt")




if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1]) # default 0
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="linear")
    parser.add_argument("--pretrained", type=int, default=1, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=3) # default 100
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu_id", type=str, default="3")

    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--save_best", type=int, default=1, choices=[0,1])

    args = parser.parse_args()
    main(args)
