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


def write_bboxes(files_list: list,
                 bboxes_list: list,
                 args: Namespace) -> None:

    data = []
    for file, bboxes in zip(files_list, bboxes_list):
        data += [[file, bbox] for bbox in bboxes]

    df = pd.DataFrame(data=data)
    df.to_csv(args.truth,
              mode='a',
              header=False,
              index=False)


def main(args: Namespace) -> None:

    print(args.files)

    ground_truth = pd.read_csv(args.truth)
    truthed = ground_truth['image_id'].unique()

    files_list = []
    bboxes_list = []
    srcs = []
    dsts = []
    images = []
    for file in os.listdir(args.data_path):

        filepath = os.path.join(args.data_path,
                                file)
        if os.path.isfile(filepath) and file not in truthed:

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

            try:
                if not bboxes:
                    bboxes = [[]]
            except ValueError:
                pass

            files_list.append(file)
            bboxes_list.append(bboxes)
            srcs.append(filepath)
            dsts.append(os.path.join(args.storage_path, file))
            images.append(image)

    write_bboxes(files_list,
                 bboxes_list,
                 args)

    for src, dst, image in zip(srcs, dsts, images):
        shutil.move(src, dst)
        cv2.imwrite(src, image)


if __name__ == '__main__':

    data_path = r'..\data'
    data_path = os.path.abspath(data_path)

    storage_path = r'..\data\originals'
    storage_path = os.path.abspath(storage_path)

    truth = r'..\data\truth\ground_truth_0.csv'
    truth = os.path.abspath(truth)

    if not os.path.isdir(data_path):
        os.mkdir(data_path)
    if not os.path.isdir(storage_path):
        os.mkdir(storage_path)
    if not os.path.isdir(os.path.dirname(truth)):
        os.mkdir(os.path.dirname(truth))
    if not os.path.isfile(truth):
        df = pd.DataFrame(columns=['image_id',
                                   'bbox'])
        df.to_csv(truth,
                  index=False)

    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path',
                        type=str,
                        help='Path to labeled images',
                        default=data_path)
    parser.add_argument('-s', '--storage_path',
                        type=str,
                        help='Path to original images',
                        default=storage_path)
    parser.add_argument('-t', '--truth',
                        type=str,
                        help='Path to ground truth .json file',
                        default=truth)
    parser.add_argument('-f', '--files',
                        nargs='+',
                        help='List of specfic files to (re)label',
                        required=False)
    args = parser.parse_args()

    main(args)
