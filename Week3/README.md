# C1-Project
### Week 2 

To be able to run the code, you need to install the requirements from the requirements.txt file. 
To do so, you need to run the following command: ```pip3 install -r requirements.txt```

In order to load the data, in main.py you need to change the path to the dataset folder depending on where you have it located.

To run the code, you need to execute the following command: ```python3 main.py```. Other scripts are not mandatory to run to obtain the pickle file.
Depending on the parameters you want to use, you need to change the values of the variables in the main.py file.
Default parameters for our model are:
- ```BASE_DESCRIPTOR = Histogram(color_model="yuv", bins=25, range=(0, 255))```
- ```SPLIT_SHAPE = (20, 20) ```
- ```DESCRIPTOR_FN = SpatialDescriptor(BASE_DESCRIPTOR, SPLIT_SHAPE)```
- ```K = 10```
- ```DISTANCE_FN = Intersection()```
- ```BG_REMOVAL_FN = RemoveBackgroundV2()```
- ```TEXT_DETECTOR = TextDetection()```

The output of the execution of the code is a pickle file with the retrieved objects from the DB for each query image.

A quick explanation of the most relevant files used is shown below:
- ```main.py``` is the main file that needs to be executed to obtain the pickle file.
- ```descriptors.py``` contains the descriptor class and the spatial descriptor class, used to compute the descriptors.
- ```distance.py``` contains all the distances classes(Intersection, ChiSquare, Hellinger, L1, L2, Cosine).
- ```retrieval.py``` contains function that returns the k most similar images for each image in queries.
- ```bg_removal.py``` contains the background removal class.
- ```text_detection.py``` contains the text detection class.
- ```utils.py``` contains the helper functions used in the code.
- ```requirements.txt``` contains the libraries needed to run the code.
- ```score_text.py``` contains the functions used to score the text detection with IoU metric.
- ```score_masks.py``` contains the functions used to score the masks.
- ```test_retrieval.py``` contains the functions used to test the retrieval system.
- ```save_*.py``` contains the functions used to save the necessary files (bbox, masks and cropped images).
