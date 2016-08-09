#Fisher Vectors

__Utility to create fisher vectors given a set of images using MR8 features__

[]()

Requirements:

OpenCV > v3.0.0
scikit-learn
Scipy 
Numpy
matplotlib

**Usage:**
```
python fisher_vectors.py [-h] [-m {loc_desc,vocab,fisher_vec}] [-o OUT]
                         input_dir

positional arguments:
  input_dir             <File path to directory with input images

optional arguments:
  -h, --help            show this help message and exit
  -m {loc_desc,vocab,fisher_vec}, --mode {loc_desc,vocab,fisher_vec}
                        <Mode to use script in : loc_desc | vocab | fisher_vec
                        >
  -o OUT, --out OUT     <Output file_name>

```


##MR8 feature creation:

Creates MR8 features as per [Link](http://www.robots.ox.ac.uk/~vgg/research/texclass/filters.html)



##Create Vocabulary:

Generates a vocabulary using a gaussian mixture model


##Create Fisher Vectors:

Uses the generated vocabulary and input image features to generate image_wise fisher vectors. 

Output of each step is a joblib pkl file with name as ```out_path_{feature_dict/vocabulary/fisher_vectors}.pkl``` which contains a dictionary with image names as a key and the corresponding feature group/ Fisher vector as output.


