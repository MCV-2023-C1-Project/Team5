# C1-Project
## Week 1

To be able to run the code, you need to install the requirements from the requirements.txt file.
To do so, you need to run the following command:
```pip3 install -r requirements.txt```

In order to load the data, in main.py you need to change the path to the dataset folder depending on where you have it located.

To run the code, you need to execute the following command:
```python3 main.py```.
Other scripts are not mandatory to run to obtain the pickle file

We haven't implemented the command line arguments yet, so you need to change the parameters in the main.py file.
In line 18 of the main.py file, you can change the number of parameters to get the results, that are automatically saved in the result.pkl file.

An example of configuration is the following:
```DESCRIPTOR_FN = Histogram(color_model="yuv", bins=25, range=(0, 255))```

The parameters are:
- color_model: the color model used to extract the histogram. It can be "rgb", "hsv", "lab", "yuv", "ycbcr"
- bins: the number of bins used to extract the histogram
- range: the range of the histogram

Code files:
- `bg_removal.py` contains definitions of classes responsible for the background removal
- `descriptors.py` contains definitions of classes responsible for the descriptors computation
- `distances.py` contains definitions of classes responsible for computing distances between 1d NumPy arrays (descriptors)
- `evaluation_funcs.py` contains functions used to numerically evaluate the quality of the retrieval system
- `main.py` produces the retrieval result, stores it as a pickle
- `optuna_search.py` wraps `main.py` into an optuna hyperparameter search
- `qualitative_test.py` displays queries and retrieval results, along with hints in the terminal
- `retrieval.py` contains the definition of the core retrieval function
- `save_masks.py` uses instances of `bg_removal.py` to create and save painting masks for a submission
- `score_masks.py` quantitatively evaluates produced masks against the ground truth
- `test_retrieval.py` quantitatively evaluates produced retrieval pickle against the ground truth
