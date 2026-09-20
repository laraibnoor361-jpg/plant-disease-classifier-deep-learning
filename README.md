# LeafCheck - Plant Disease Detector

Deep learning app (MobileNetV2 transfer learning) built with TensorFlow and Streamlit.
Add a photo of a leaf and it names the plant, the disease and what to do next.

## Files
| File | What it does |
|---|---|
| `app.py` | The Streamlit web app (UI + prediction) |
| `train.py` | Trains the model, tests it and writes the `models/` folder |
| `make_samples.py` | Copies a few demo photos into `sample_images/` |
| `requirements.txt` | Packages Streamlit Cloud installs |

## Run
```
pip install -r requirements.txt
python train.py --data_dir "path/to/train" --epochs 5 --fine_tune_epochs 3 --max_per_class 400
python make_samples.py --data_dir "path/to/train"
streamlit run app.py
```

Dataset: PlantVillage-style leaf photos, one folder per class (for example the Kaggle
"New Plant Diseases Dataset (Augmented)").
