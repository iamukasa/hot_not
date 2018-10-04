

<div align='center'>
<img src = 'irvnin-01.png' height="200px">
</div>

## Building our own model

Creating a CNN from Scratch
* [Colab code](https://colab.research.google.com/drive/10XIGvbdXPX9525yedCWDvYterIYYhbKY)



## Image Classifier through transfer learning

A Simple transfer learning with an Inception v3 and Imagenet architecture model which
displays summaries in TensorBoard.

This example shows how to take a Inception v3 architecture model trained on
ImageNet images, and train a new top layer that can recognize other classes of
images.

The top layer receives as input a 2048-dimensional vector for each image. We
train a softmax layer on top of this representation. Assuming the softmax layer
contains N labels, this corresponds to learning N + 2048*N model parameters
corresponding to the learned biases and weights.


## Requirements

* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [TensorFlow 1.7](https://www.tensorflow.org/install/)
* [SciPy & NumPy](http://scipy.org/install.html)
* [Kaggle hot dog not hotdog  dataset](https://drive.google.com/drive/folders/1y-nVLx4tGrdSohdlbYOgAfWoB7uWy3Fi?usp=sharing)


## Dependencies
 To install dependenxies you need to run

```bash
pip install -r requirements.txt
```


##Example

Here's an example, which assumes you have a folder containing class-named
subfolders, each full of images for each label. The example folder flower_photos
should have a structure like this:

~/data/hot_dog/photo1.jpg
~/data/hot_dog/photo2.jpg
...
~/data/not_hot_dog/anotherphoto77.jpg
~/data/not_hot_dog/somepicture.jpg
...

The subfolder names are important, since they define what label is applied to
each image, but the filenames themselves don't matter. Once your images are
prepared, you can run the training with a command like this:

```bash
python retrain.py --model_dir ./inception --image_dir data
```

## Visualisation
To use with TensorBoard:

By default, this script will log summaries to /tmp/retrain_logs directory

Visualize the summaries with this command:

Inception

```bash
tensorboard --logdir /ouput/retrain_logs

```

mobilenets
```bash
tensorboard --logdir /tmp/retrain_logs

```

## Testing

Inception
```bash
python retrain_model_classifier.py data/hot_dog/3742819.jpg
```
mobilenets


python label_image.py --image=data/hot_dog/3742819.jpg --graph=output/retrained_graph.pb --labels=output/retrained_labels.txt





## retraining mobilenets
python retrain.py \
  --bottleneck_dir=output/bottlenecks \
  --how_many_training_steps=4000 \
  --output_graph=output/retrained_graph.pb \
  --output_labels=output/retrained_labels.txt \
  --image_dir=data

##retraning inception
sudo python retrain.py --image_dir data ./Inception 

## Converting to tflite inception

toco \
  --input_file=ouput/output_graph.pb \
  --output_file=ouput/output_graph.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shape=1,2048,2048,3 \
  --input_array=input \
  --output_array=final_result \
  --inference_type=FLOAT \
  --input_data_type=FLOAT\
  --allow_custom_ops

## Converting to tflite mobilenets
``` bash
toco \
  --input_file=output/retrained_graph.pb \
  --output_file=output/output_graph.tflite \
  --input_format=TENSORFLOW_GRAPHDEF \
  --output_format=TFLITE \
  --input_shape=1,224,224,3 \
  --input_array=input \
  --output_array=final_result \
  --inference_type=FLOAT \
  --input_data_type=FLOAT\
  --allow_custom_ops

```
