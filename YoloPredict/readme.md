# SETUP

- Upload dataset with labels and images to YoloPredict folder. Structure of dictionary:  
```
dataset/
├── train/
│   ├── images/      # Training images
│   └── labels/      # Corresponding labels for training images
├── val/
│   ├── images/      # Validation images
│   └── labels/      # Corresponding labels for validation images
```

- Set path in data.yaml to folders train and val
- Set hyperparameters in YoloPredict/constants.py
- Run model.py