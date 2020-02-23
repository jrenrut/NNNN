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


A workflow is available to create new labeled data. Save images to `data/`. The script `util/labeler.py` uses `OpenCV` to allow the user to select bounding rectangle around any / all / no targets in the images. Images are then downsampled such that max(height, width) = 500 and updated within `data/`. Bounding rectangle information is saved to `data/truth/ground_truth_0.csv`, and original images are saved to `data/originals/`.

A small sample of true negative images are included in `data/`, taken from the PASCAL Visual Object Classes Challenge.

To label new data, simply save the images to `data/` and run:

```bash
cd util
python labeler.py
```

This will only offer images not in `data/truth/ground_truth_0.csv` for bounding rectangle selection.

To re-label data or label a specific new image (or both), pass the `-f` argument with the file names:

```bash
cd util
python labeler.py -f 0.jpeg 3.jpeg
```

## TODO:

- [ ] Do the actual work of building / training model.
