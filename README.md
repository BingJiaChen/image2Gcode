# EECamp 2021 Minidrawing Machine - image2Gcode

## Description
With this code, one can turn image into gcode file, so that minidrawing machine can work on it

## Turtorial
Please follow google colab link:
[link](https://drive.google.com/file/d/1M-FZAz9-J8OuBllEiKsv-XR5LVi2fFV0/view?usp=sharing)

## Usage (local)

Please install below packages in advance
- [pypotrace](https://pypi.org/project/pypotrace/)
- [matplotlib](https://pypi.org/project/matplotlib/)
- [numpy](https://pypi.org/project/numpy/)
- [scipy](https://pypi.org/project/scipy/)

Image to gcode
```bash
$ python genGcode.py
# output will be saved in /out directory
```

Upload gcode to Arduino
```bash
$ python GcodeSender.py
# make sure the PORT is the same as your environment (COM4 for example)
```

## Contributors
[BingJiaChen](https://github.com/BingJiaChen)

[Ken Chu](https://github.com/Kenchu123)

