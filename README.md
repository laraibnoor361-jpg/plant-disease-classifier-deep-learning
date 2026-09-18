# Plant Disease Image Classifier

## Project Overview
This is a simple student-level deep learning project that classifies
plant leaf images into disease categories using **transfer learning**
with **MobileNetV2**. It includes a training script (`train.py`) and a
Streamlit app (`app.py`) for uploading a leaf photo and getting a
prediction.

## Dataset: PlantVillage
This project uses the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease),
which contains thousands of leaf images across many crop species and
diseases (for example `Apple___Apple_scab`, `Tomato___healthy`,
`Potato___Late_blight`, etc.).

The dataset is **not included** in this project folder because it is
too large. To use it:

1. Download the PlantVillage dataset (for example the "color" version
   from Kaggle).
2. Place the class folders directly inside the `dataset/` folder, like
   this:

   ```
   dataset/
       Apple___Apple_scab/
           image1.jpg
           image2.jpg
           ...
       Apple___healthy/
           ...
       Tomato___healthy/
           ...
   ```

3. Each sub-folder name becomes a class name automatically - `train.py`
   reads the folder names directly, so **no class names are
   hard-coded** anywhere in the code.

## MobileNetV2 (Transfer Learning)
Instead of building and training a CNN from scratch (which needs a lot
of data and time), this project reuses **MobileNetV2**, a lightweight
model already pretrained on the large ImageNet dataset. The pretrained
base layers are **frozen** (not retrained), and a small classification
head (a couple of dense layers) is added on top and trained on the
PlantVillage images. This is a common and efficient way to build image
classifiers with limited data and compute.

## Training Process (`train.py`)
Running `train.py` will:

1. Scan the `dataset/` folder and report the number of images, number
   of classes, class names, and images per class.
2. Load the images using Keras's `ImageDataGenerator`, resizing them to
   224x224 (the size MobileNetV2 expects), applying MobileNetV2's
   preprocessing, and splitting the data into training and validation
   sets (80% / 20%).
3. Build a MobileNetV2-based model with a frozen base and a simple
   classification head.
4. Train the model and print training/validation accuracy and loss.
5. Save accuracy/loss charts to `models/training_history.png`.
6. Show sample predictions on a few validation images and save them to
   `models/sample_predictions.png`.
7. Save the trained model to `models/plant_disease_model.keras` and
   the class names to `models/class_names.json`, so the Streamlit app
   always uses the exact same classes and order used in training.

## Streamlit App (`app.py`)
The app loads the trained model and class names, lets you upload a
leaf image, shows the image, and displays:
- The predicted class name
- The prediction confidence (%)
- The top 3 most likely classes

If no trained model is found yet, the app will show a message asking
you to run `train.py` first.

## Technologies
- Python
- TensorFlow / Keras
- MobileNetV2
- NumPy
- Matplotlib
- Pillow
- Streamlit

## How to Install Requirements

```bash
pip install -r requirements.txt
```

## How to Train the Model

1. Add your PlantVillage images inside `dataset/` (see the Dataset
   section above).
2. Run:

```bash
python train.py
```

This creates the `models/` folder with the trained model, the class
names file, and the accuracy/loss and sample-prediction charts.

## How to Run the Streamlit App

After training is finished:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal, upload a leaf image,
and click **Predict**.
