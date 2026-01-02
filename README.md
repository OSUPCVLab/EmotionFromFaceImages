## MotivNet: Evolving Meta-Sapiens into an emotionally intelligent model.

MotivNet is a model built on Meta-Sapiens and ML-Decoder for the FER task on AffectNet. <a href="https://arxiv.org/abs/2512.24231">See Paper</a>

We provide code to run inference on the model and code to train the model on your own data with custom specifications.

For inference, the output labels are shown below 

0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger.

### Get Started

Getting started is very simple. 

First, create a new virtual environment with and run <code>pip install -r requirements.txt</code>

Then, download the MotivNet checkpoint from OneDrive. [https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/medicharla_2_buckeyemail_osu_edu/EfnsSxS42JNDipAEU45o-bUBsGfXviOOgaWka5LBLBkvBA?e=1SF6v8](https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/medicharla_2_buckeyemail_osu_edu/IQD5Yo67QSPXQJluT1mCEZ4uAbm9hxWWIC1bXf7l43S9X3w?e=H7vRd4)

Place this checkpoint in the /checkpoints/ folder to start finetuning or running inference on the model (defined in <code>model.py</code>)

### Inference

To run predictions on a set of images, put all of the images in one folder and use the <code>inference.py</code> file to run your predictions

### Train

To train MotivNet for your own custom use case, you need 4 values to pass into the <code>train.py</code> file

1. <code>--train_data</code> This value is the folder path for all the images that comprise your training dataset
2. <code>--train_labels</code>This value holdes the file path for the ground truths of your training dataset. It is a JSON file where each key is the emotion_label (0,1,...,7), as defined above, and it holds an array of the image file names belonging to that emotion class. 
3. <code>--test_data</code> This value is the folder path for all the images that comprise your test dataset
4. <code>--test_labels</code>This value holdes the file path for the ground truths of your tests dataset. It's stored in the same format as the train_labels. 