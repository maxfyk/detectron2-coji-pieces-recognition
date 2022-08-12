##############################################################################
#
# Below code is inspired on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/pascal_voc.py
# https://github.com/jagin/detectron2-licenseplates
# --------------------------------------------------------
# Detectron2
# Licensed under the Apache 2.0 license.
# --------------------------------------------------------
import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch

from coji.dataset import register_coji_voc
from coji.config import setup_cfg
from coji.trainer import Trainer


def main(args):
    # Register coji dataset
    register_coji_voc("coji_train", "datasets/coji", "train")
    register_coji_voc("coji_test", "datasets/coji", "test")

    # Setup model configuration
    cfg = setup_cfg(args)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    # Run training process
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
