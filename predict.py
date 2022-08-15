##############################################################################
#
# Below code is inspired on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/pascal_voc.py
# https://github.com/jagin/detectron2-licenseplates
# --------------------------------------------------------
# Detectron2
# Licensed under the Apache 2.0 license.
# --------------------------------------------------------

import random
import cv2
from PIL import Image
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog

from coji.dataset import register_coji_voc
from coji.config import setup_cfg


def parse_args():
    import argparse

    # Parse command line arguments
    ap = argparse.ArgumentParser(description="COJI pieces recognition")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--scale", type=float, default=1.0)

    # Detectron settings
    ap.add_argument("--config-file",
                    required=True,
                    help="path to config file")
    ap.add_argument("--confidence-threshold", type=float, default=0.6,
                    help="minimum score for instance predictions to be shown (default: 0.5)")
    ap.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                    help="modify model config options using the command-line")

    return ap.parse_args()


def main(args):
    if args.confidence_threshold is not None:
        # Set score_threshold for builtin models
        args.opts.append('MODEL.ROI_HEADS.SCORE_THRESH_TEST')
        args.opts.append(str(args.confidence_threshold))
        args.opts.append('MODEL.RETINANET.SCORE_THRESH_TEST')
        args.opts.append(str(args.confidence_threshold))

    dataset_name = "geom-original-test"
    register_coji_voc(dataset_name, "datasets/geom-original", "test")
    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    img = cv2.imread("C:\\Users\\maxfyk\\Downloads\\photo_2022-08-12_16-24-41.jpg")
    prediction = predictor(img)
    visualizer = Visualizer(img[:, :, ::-1],
                            metadata=MetadataCatalog.get(dataset_name),
                            scale=args.scale)
    vis = visualizer.draw_instance_predictions(prediction["instances"].to("cpu"))
    img = vis.get_image()[:, :, ::-1]
    im_pil = Image.fromarray(img)
    im_pil.show()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    print("Command Line Args:", args)

    main(args)
