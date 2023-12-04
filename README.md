# inf721-tpfinal
This is the final project of INF 721 (Learning in Deep Neural Networks).
The model was trained using [Pytorch v2.0.1](https://pytorch.org/) with an online dataset for kitchen utensils classification [Kitchen Utensils](https://homepages.inf.ed.ac.uk/rbf/UTENSILS/kitchen_utensils1.xml).

## Directory structure
- 'scripts': Has all scripts used to train the model
- 'ObjectDetection': Has all the source code used for creating the application to run the model on android devices

## Tutorials
* obs: if you coudn't select a cuda device for training, even though your machine has one, try running the following line, notice tha it will override the current torch module instaled on our machine:
```console
pip install torch==2.0.1+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- Training: run the script 'scripts/inf721_train.py'. The trained model will be saved on the file 'last-run.pth' with useful information for resuming training. Notice the defined number of epochs is 50, if you plan to train for more epochs, go the the line 70 of the file and change it.
- Inference: run the script 'scripts/inf721_inference.py'. The trained model will be loaded from the file 'last-run.pth', so make sure to have an valid file trained using the training script.

If you prefer, you can try running the notebook 'Object_detection_model.ipynb' on [Colab](https://colab.research.google.com/github/johnpolsh/inf721-tpfinal/blob/main/colab/Object_detection_model.ipynb).
