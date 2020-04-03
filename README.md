## CNN models

We implemented some CNN models and analyzed their prediction power for a computer vision object classification task on the fashion-mnist dataset (see https://github.com/zalandoresearch/fashion-mnist). 
We analysed the following models:

- CNN model with 1 convolutional layer followed by a max pooling layer and 2 fully connected layers (with dropout).
For specific model details see the following class implementations in main.py:
  - CNN1pooling1fully2_a
  - CNN1pooling1fully2_b
  - CNN1pooling1fully2_c
  - CNN1pooling1fully2_d

- CNN model with 2 convolutional layers followed by max pooling layers and 2 fully connected layers (with dropout). For specific model details see the following class implementations in main.py
  - CNN2pooling2fully2_a
  - CNN2pooling2fully2_b
  - CNN2pooling2fully2_c
  - CNN2pooling2fully2_d
  
## Results comparison

| Model        | Training accuracy           | Test accuracy  |  Number of parameters  |
| :------------ |:--------------| :-----| :-----|
| CNN1pooling1fully2_a  |  96.91%  |  90.88%  | 1.617k |
| CNN1pooling1fully2_b  |  90.48%  |  87.42%  | 602k |
| CNN1pooling1fully2_c  |  95.70%  |  91.38%  | 405k |
| CNN1pooling1fully2_d  |  93.00%  |  88.22%  | 151k |
| CNN2pooling2fully2_a  |  94.02%  |  90.74%  | 110k |
| CNN2pooling2fully2_b  |  91.26%  |  87.28%  | 78k |
| CNN2pooling2fully2_c  |  93.62%  |  90.64%  | 53k |
| CNN2pooling2fully2_d  |  89.89%  |  87.26%  | 45k |

We see that the majority of the tested CNNs have less parameters then the benchmarks mentioned at https://github.com/zalandoresearch/fashion-mnist
Our training results indicate, based on the *test accuracy* and the *number of parameters*, that the *CNN2pooling2fully2_c* model might be the most appropriate choice for our task if a speed-accuracy tradeoff is preferd versus pure model accuracy.
  
## Running the code

In order to run the code the user requires tensorflow 2 or tensorflow 2.1 (code has been tested on tensorflow 2.1).

The code can be run from the console. For training we can run, e.g., 
```console
foo@mnist:~$ python main.py --device CPU --action train --model CNN2pooling2fully2_a
```
 
For prediction based on a pretrained model we can run, e.g., 
```console
foo@mnist:~$ python main.py --action predict --model_dir CNN2pooling2fully2_a\logs\checkpoints_20200403-055721
``` 

Visualization of the training process can be viewed in tensorboard by running, e.g., 
```console
foo@mnist:~$ tensorboard --logdir CNN2pooling2fully2_a\logs\tb_20200403-055721
``` 
Note that the intersections between the training and validation accuracy/loss lines indicate that the prediction power of the analyzed CNN will not increase by training on further epochs.

 



 
