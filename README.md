## MotivNet: Evolving Meta-Sapiens into an emotionally intelligent model.

MotivNet is a model built on Meta-Sapiens and ML-Decoder for the FER task on AffectNet.

We provide code to run inference on the model and code to train the model on your own data with custom specifications.

For inference, the output labels are shown below 

0: Neutral, 1: Happiness, 2: Sadness, 3: Surprise, 4: Fear, 5: Disgust, 6: Anger,
7: Contempt.

### Get Started

Getting started is very simple. 

First, create a new conda environment and run <code>pip install -r requirements.txt</code>

Then, download two checkpoints. The first is MotivNet.pth from OneDrive. These are the weights associated with MotivNet. https://buckeyemailosu-my.sharepoint.com/:u:/g/personal/medicharla_2_buckeyemail_osu_edu/EfnsSxS42JNDipAEU45o-bUBsGfXviOOgaWka5LBLBkvBA?e=1SF6v8

The second is Sapiens 1B parameter pose estimation model from hugging face. https://huggingface.co/facebook/sapiens-pose-1b-torchscript/tree/main

Place both of these checkpoints in a new checkpoints folder and get started!



