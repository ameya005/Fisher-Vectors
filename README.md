#Fisher Vectors

__Utility to create fisher vectors given a set of images using MR8 features__

[]()

Requirements:

OpenCV > v3.0.0
scikit-learn
Scipy 
Numpy
matplotlib


MR8 feature creation:

Creates MR8 features as per [Link](http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html)

Usage: 
```
python fisher_vectors.py <Path to directory with images> --create_loc_desc
```


Create Vocabulary:

Generates a vocabulary using a gaussian mixture model

Usage: 
```
python fisher_vectors.py <Path to directory with images> --create_vocab
```

Create Fisher Vectors:

Uses the generated vocabulary and input image features to generate image_wise fisher vectors. 

Usage:
```
python fisher_vectors.py <Path_to_directory_with_images> --create_fisher_vec 
``` 


