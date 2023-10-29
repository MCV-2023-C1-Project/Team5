# C1-Project
### Week 2 

To be able to run the code, you need to install the requirements from the requirements.txt file. 
To do so, you need to run the following command: ```pip3 install -r requirements.txt```

In order to load the data, in main.py you need to change the path to the dataset folder depending on where you have it located.

To run the code, you need to execute the following command: ```python3 main.py```. Other scripts are not mandatory to run to obtain the pickle file.

You should update the paths to be equal to your local paths.
```QUERY_IMG_DIR = Path(os.path.join("data", "Week3", "qsd2_w3"))```
```REF_IMG_DIR = Path(os.path.join("data", "Week1", "BBDD"))```
```RESULT_OUT_PATH = Path(os.path.join("results", "color_text_qsd2_K1.pkl"))```

There is a boolean to decide if use Text combination for reordering the results from color/texture or both.
```with_text_combination = True```

Depending on the parameters you want to use, you need to change the values of the variables in the main.py file.
For using different descriptors you should comment the not wanted descriptors and use only the one you want to use.
Default parameters for our model are:
```BASE_DESCRIPTOR = Histogram(color_model="yuv", bins=25, range=(0, 255))```
```# BASE_DESCRIPTOR = LocalBinaryPattern(numPoints=8, radius=1)```
```SPLIT_SHAPE = (20, 20)  # (1, 1) is the same as not making spatial at all```
```DESCRIPTOR_FN = SpatialDescriptor(BASE_DESCRIPTOR, SPLIT_SHAPE)```
```K = 1```
```DISTANCE_FN = Intersection()```
```NOISE_FILTER = Median()```
```NAME_FILTER = Average()```
```TEXT_DETECTOR = TextDetection()```
```HAS_NOISE = Salt_Pepper_Noise(noise_filter=NOISE_FILTER,```
                              ```name_filter=NAME_FILTER,```
                              ```text_detector=TEXT_DETECTOR)```


The output of the execution of the code is a pickle file with the retrieved objects from the DB for each query image.

A quick explanation of the most relevant files used is shown below:
- ```main.py``` is the main file that needs to be executed to obtain the pickle file.
- ```descriptors.py``` contains the color, texture, spatial, text descriptor classes.
- ```distance.py``` contains all the distances classes(Intersection, ChiSquare, Hellinger, L1, L2, Cosine).
- ```retrieval.py``` contains function that returns the k most similar images for each image in queries.
- ```bg_removal.py``` contains the background removal class.
- ```noise_removal.py``` contains the noise removal class.
- ```filters.py``` contains the filters used to remove noise.
- ```text_detection.py``` contains the text detection class.
- ```text_combination.py``` contains the text combination (to reorder the results) class.
- ```utils.py``` contains the helper functions used in the code.
- ```requirements.txt``` contains the libraries needed to run the code.
- ```score_text.py``` contains the functions used to score the text detection with IoU metric.
- ```score_masks.py``` contains the functions used to score the masks.
- ```test_retrieval.py``` contains the functions used to test the retrieval system.
- ```save_*.py``` contains the functions used to save the necessary files (bbox, masks and cropped images).
- ```paintings_db_bbdd.csv``` is a CSV file containg the authors names from all the BBDD's paintings.
