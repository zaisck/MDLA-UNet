# MDLA-UNet

**0. Main Environments**
- python 3.8
- pytorch 1.8.0
- torchvision 0.9.0

**1. Prepare the dataset.**

- The ISIC17 and ISIC18 datasets, divided into a 7:3 ratio. You are supposed to put them into './data/isic17/' and './data/isic18/', and the file format reference is as follows. (take the ISIC17 dataset as an example.)

- './data/isic17/'
  - train
    - images
      - .png
    - masks
      - .png
  - val
    - images
      - .png
    - masks
      - .png

**2. Train the MDLA-UNet.**
```
cd MDLA-UNet
```
```
python train.py
```

**3. Obtain the outputs.**
- After trianing, you could obtain the outputs in './results/'
