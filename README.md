# fashion-mnist CNN-approach

Analyzing performance of some CNN architectures on the MNIST-fashion dataset.

The project is based on the data available at https://github.com/zalandoresearch/fashion-mnist

## CNN models

We analysed the following models:

- CNN model with 1 convolutional layers followed by a max pooling layer and a final fully connected layer (with dropout).
For model details see the following class implementations in main.py:
  - CNN1pooling1fully1_a
  - CNN1pooling1fully1_b
  - CNN1pooling1fully1_c
  - CNN1pooling1fully1_d

- CNN model with 2 convolutional layers followed by max pooling layers and a final fully connected layer (with dropout). For model details see the following class implementations in main.py
  - CNN2pooling2fully1_a
  - CNN2pooling2fully1_b
  - CNN2pooling2fully1_c
  - CNN2pooling2fully1_d
  
## Results comparison

| Model        | Training accuracy           | Test accuracy  |  Number of parameters  |
| :------------ |:--------------| :-----| :-----|
| CNN1pooling1fully1_a  |  96.91%  |  90.88%  | 1.617k |
| CNN1pooling1fully1_b  |  90.48%  |  87.42%  | 602k |
| CNN1pooling1fully1_c  |  95.70%  |  91.38%  | 405k |
| CNN1pooling1fully1_d  |  93.00%  |  88.22%  | 151k |
| CNN2pooling2fully1_a  |  94.02%  |  90.74%  | 110k |
| CNN2pooling2fully1_b  |  91.26%  |  87.28%  | 78k |
| CNN2pooling2fully1_c  |  93.62%  |  90.64%  | 53k |
| CNN2pooling2fully1_d  |  89.89%  |  87.26%  | 45k |

We see that the majority of the tested CNN have less parameters then the benchmarks mentioned at https://github.com/zalandoresearch/fashion-mnist
Our training results indicate, based on the *test accuracy* and the *number of parameters* that the *CNN2pooling2fully1_c* model might be the most appropriate selection for our task if the speed of the model evaluation is important.
  
## Running the code

The code can be run from the console. For training we can run, e.g., 
```console
foo@mnist:~$ python main.py --device CPU --action train --model CNN2pooling2fully1_a
```
 
For prediction based on a predetermined model we can run, e.g., 
```console
foo@mnist:~$ python main.py --action predict --model_dir CNN2pooling2fully1_a\logs\checkpoints_20200403-055721
``` 

Visualization of the training process can be viewed in tensorboard by running, e.g., 
```console
foo@mnist:~$ tensorboard --logdir CNN2pooling2fully1_a\logs\tb_20200403-055721
``` 

 



 
