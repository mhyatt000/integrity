# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.md
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt

import argparse
import glob
import os
import time

from PIL import Image
from torch.cuda import is_available as is_available_cuda

from resnet50nodown import R50nodown

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This script tests the network on an image folder and collects the results in a CSV file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights_path",
        "-m",
        type=str,
        default="./weights/gandetection_resnet50nodown_stylegan2.pth",
        help="weights path of the network",
    )
    parser.add_argument(
        "--input_folder",
        "-i",
        type=str,
        default="./example_images",
        help="input folder with PNG and JPEG images",
    )
    parser.add_argument("--output_csv", "-o", type=str, default=None, help="output CSV file")
    args = parser.parse_args()

    device = "cuda:0" if is_available_cuda() else "cpu"
    net = R50nodown(device, args.weights_path)
    output_csv = args.output_csv or f"out.{os.path.basename(args.input_folder)}.csv"

    list_files = sorted(
        sum(
            [
                glob.glob(os.path.join(args.input_folder, "*." + x))
                for x in ["jpg", "JPG", "jpeg", "JPEG", "png", "PNG"]
            ],
            list(),
        )
    )
    num_files = len(list_files)

    print("GAN IMAGE DETECTION")
    print("START")

    with open(args.output_csv, "w") as fid:
        fid.write("filename,logit,time\n")
        fid.flush()

        for index, filename in enumerate(list_files):
            print("%5d/%d" % (index, num_files), end="\r")

            tic = time.time()
            img = Image.open(filename).convert("RGB")
            img.load()
            logit = net.apply(img)
            toc = time.time()

            fid.write("%s,%f,%f\n" % (filename, logit, toc - tic))
            fid.flush()

    print("\nDONE")
    print("OUTPUT: %s" % args.output_csv)
