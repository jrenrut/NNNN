# NNNN
**N**erf or **N**othin' **N**eural **N**etwork

<p align="center"> 
<img src="https://i.imgur.com/MFjRp4F.png">
</p>

The universal binary image classifier.

## Setup

With conda (4.8.2):
```bash
conda env create -f env.yml
conda activate NNNN
```

## Making Ground Truth

Images are saved as `jpg`s, downsampled such that max(height, width) = 500.

A small sample of true negative images are saved in `data/pascal_voc/` (images taken from the PASCAL Visual Object Classes Challenge).

A workflow is available to create new labeled data. Save images to `data/nerf_unlabeled/`. The script `util/labeler.py` uses `OpenCV` to allow the user to select bounding rectangle around any / all targets in the images within `data/nerf_unlabeled/`. Images are then downsampled and saved to `data/nerf_labeled/`, label information is saved to `data/truth/ground_truth_0.csv`, and original images are saved to `data/nerf_unlabeled_orig/`.

```bash
cd util
python labeler.py
```

## TODO:

- [ ] More robust labeling
  - [ ] Make more generic folder names to include TP and TN images
  - [ ] Ensure no double names / rows in ground truth csv
  - [ ] Use config file rather than hard-coding paths
- [ ] Do the actual work of building / training model.
