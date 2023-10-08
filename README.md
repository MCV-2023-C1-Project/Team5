# C1-Project
1. Week 1
To be able to run the code, you need to install the following libraries:
- numpy
- PIL
- pathlib
- tqdm
- pickle
- cv2
- scipy
- pandas
- matplotlib

In order to run the code, you need to run the following command:
```python3 main.py```
We haven't implemented the command line arguments yet, so you need to change the parameters in the main.py file.
In line 18 of the main.py file, you can change the number of parameters to get the results, that are automatically saved in the result.pkl file.

An example of configuration is the following:
```DESCRIPTOR_FN = Histogram(color_model="yuv", bins=25, range=(0, 255))```

The parameters are:
- color_model: the color model used to extract the histogram. It can be "rgb", "hsv", "lab", "yuv", "ycbcr"
- bins: the number of bins used to extract the histogram
- range: the range of the histogram
