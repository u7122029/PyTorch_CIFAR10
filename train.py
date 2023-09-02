import os
from argparse import ArgumentParser

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path

from data import MNISTData, CIFAR10Data
from module import CIFAR10Module, all_classifiers


def main(args):

    if bool(args.download_weights):
        CIFAR10Data.download_weights()
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    checkpoint = ModelCheckpoint(
        dirpath=f"checkpoints/{args.task}/{args.classifier}",
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
    if args.task == "cifar10":
        data = CIFAR10Data(args)
    else:
        data = MNISTData(args)

    model = CIFAR10Module(vars(args))
    if bool(args.pretrained):
        state_dict = os.path.join(
            "cifar10_models", "state_dicts", args.task, args.classifier + ".pt"
        )
        model.model.load_state_dict(torch.load(state_dict))
        trainer.test(model, data.test_dataloader())
        return

    trainer.fit(model, data.train_dataloader(), data.test_dataloader())
     # validation dataset is the same as the testing dataset. No need to test.

    if args.save_best:
        # Load best checkpoint
        loaded = CIFAR10Module.load_from_checkpoint(f"checkpoints/{args.task}/{args.classifier}/best.ckpt", hparams=vars(args))
        save_path = Path("cifar10_models") / "state_dicts" / args.task
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save(loaded.model.state_dict(), str(save_path / f"{args.classifier}.pt"))


if __name__ == "__main__":
    parser = ArgumentParser()

    # PROGRAM level args
    parser.add_argument("--data_dir", type=str, default="/ml_datasets")
    parser.add_argument("--task", type=str, default="mnist", choices=["cifar10", "mnist"])
    parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
    parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1]) # default 0
    parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"]
    )

    # TRAINER args
    parser.add_argument("--classifier", type=str, default="mobilenet_v2", choices=all_classifiers)
    parser.add_argument("--pretrained", type=int, default=1, choices=[0, 1])

    parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10) # default 100
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--gpu_id", type=str, default="3")

    parser.add_argument("--learning_rate", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)

    parser.add_argument("--save_best", type=int, default=1, choices=[0,1])

    args = parser.parse_args()
    main(args)
