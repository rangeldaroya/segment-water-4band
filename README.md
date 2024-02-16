Final results [here](https://drive.google.com/drive/folders/1Bw9MdeUwjlVLpP1C0h_4hDXbLo-H1XEG?usp=drive_link)
Takes 4 bands as input, and outputs water mask.

# Training the Model
1. Download `segtrain_data.zip` and unzip to `segtrain_data` folder that contains training data from [here](https://drive.google.com/file/d/18HCXhSLyRXisK3F9091QnMONSJ8yMIZq/view?usp=drive_link)
2. Run the following to run the model training:
``` python train_seg.py --model_type "deeplab_mobilenet" --input_type "4band"```

# Running the model on new samples
1. Download pre-trained model [here](https://drive.google.com/file/d/1mA1xyg8h1pxWnBH53bAadadZxMw-zGxM/view?usp=drive_link)
2. Download sample data [here](https://drive.google.com/file/d/13tCaFVbkpGvNJxK-N9mslrQeaAn2QqZZ/view?usp=drive_link)
3. Run the following command.
```python eval_model.py --fp <path-to-tif-file>```
or 
```python eval_model.py --fp Connecticut_20230706_01.tif```
4. The predictions will be saved in the same folder as the input tif file. There will be three output files:
    - "*_pred.png" full size png of the predicted mask (takes longer to load)
    - "*_pred.png.jpg" smaller version of the predicted mask (you'd open this if you just want a quick peek at the output)
    - "*.tif" tif of the predicted mask