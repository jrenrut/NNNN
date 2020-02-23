import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil
from argparse import ArgumentParser, Namespace


def get_bboxes(image: np.ndarray) -> list:

    winname = '<enter/space> to select, <c> to cancel, <esc> to quit'
    bboxes = cv2.selectROIs(winname,
                            image)
    cv2.destroyWindow(winname)

    for i, bbox in enumerate(bboxes):

        x, y, dx, dy = bbox
        plt.figure()
        plt.imshow(image[y:y + dy, x:x + dx][:, :, ::-1])
        plt.show()

    good = input('Use Selection(s)? [y]/n: ').lower()

    if good == 'y':
        return bboxes
    else:
        return get_bboxes(image)


def write_bboxes(file: str,
                 bboxes: list,
                 args: Namespace) -> None:

    if not os.path.isfile(args.truth):
        df = pd.DataFrame(columns=['image_id',
                                   'bbox'])
        df.to_csv(args.truth)

    data = [[file, bbox] for bbox in bboxes]
    df = pd.DataFrame(data=data)
    df.to_csv(args.truth,
              mode='a',
              header=False)

    src = os.path.join(args.unlabeled,
                       file)
    dst = os.path.join(args.unlabeled_orig,
                       file)
    shutil.move(src, dst)


def main(args: Namespace) -> None:

    for file in os.listdir(args.unlabeled):

        filepath = os.path.join(os.path.abspath(args.unlabeled),
                                file)

        image = cv2.imread(filepath)

        width, height, _ = image.shape
        scale = float(500 / max(width, height))
        width = int(np.ceil(width * scale))
        height = int(np.ceil(height * scale))

        image = cv2.resize(image,
                           (2*height, 2*width),
                           interpolation=cv2.INTER_AREA)

        bboxes = get_bboxes(image)

        image = cv2.resize(image,
                           (height, width),
                           interpolation=cv2.INTER_AREA)

        fig, ax = plt.subplots(1)
        ax.imshow(image[:, :, ::-1])
        for i, bbox in enumerate(bboxes):
            x, y, dx, dy = bbox
            bboxes[i] = [int(x/2),
                         int(y/2),
                         int(x/2) + int(dx/2),
                         int(y/2) + int(dy/2)]
            rect = patches.Rectangle((x/2, y/2),
                                     dx/2,
                                     dy/2,
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
        plt.show()
        print('Image labeled')

        write_bboxes(file,
                     bboxes,
                     args)

        src = os.path.join(args.labeled,
                           file)
        cv2.imwrite(src, image)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--labeled',
                        type=str,
                        help='Path to labeled images',
                        default=r'..\data\nerf_labeled')
    parser.add_argument('--unlabeled',
                        type=str,
                        help='Path to unlabeled images',
                        default=r'..\data\nerf_unlabeled')
    parser.add_argument('--unlabeled_orig',
                        type=str,
                        help='Path to original unlabeled images',
                        default=r'..\data\nerf_unlabeled_orig')
    parser.add_argument('--truth',
                        type=str,
                        help='Path to ground truth .json file',
                        default=r'..\data\truth\ground_truth_0.csv')
    args = parser.parse_args()

    main(args)
