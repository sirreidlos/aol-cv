# ACF impl

0. Clone repository and install dependencies 
```bash
git clone git@github.com:sirreidlos/aol-cv.git
cd aol-cv
# I recommend making a venv beforehand
pip install -r requirements.txt # or uv sync
```

1. Download the WIDER FACE dataset
[WIDER FACE](https://shuoyang1213.me/WIDERFACE/)

Download all three images and the face annotation

```bash
mkdir data
unzip WIDER_train.zip -d data
unzip WIDER_val.zip -d data
unzip WIDER_test.zip -d data
unzip wider_face_split.zip -d data
```

Ensure that the folder structure looks like this:
```
    ┌── images            
  ┌─┴ WIDER_val           
  │ ┌── images            
  ├─┴ WIDER_train         
  │ ┌── images            
  ├─┴ WIDER_test          
  │ ┌── wider_face_train_bbx_gt.txt            
  ├─┴ wider_face_split
┌─┴ data                  
```

2. Run training

Example:
```bash
python train.py --annotation_file data/wider_face_split/wider_face_train_bbx_gt.txt --image_dir data/WIDER_train/images/ --val_annotation_file data/wider_face_split/wider_face_val_bbx_gt.txt --val_image_dir data/WIDER_val/images/ --hidden_sizes 512 256 128 64
```

3. Run inference

Example:
```bash
python inference.py --model models/acf_detector.pkl --image data/WIDER_test/images/0--Parade/0_Parade_marchingband_1_9.jpg --output out.jpg --score_threshold 0.9
```

4. Run evaluation
Example:
```bash
python evaluate.py --model models/acf_detector.pkl --annotation_file data/wider_face_split/wider_face_val_bbx_gt.txt --image_dir data/WIDER_val/images/ --max_images 100
```

5. Run evaluation to compute PR curve and AP
```bash
python evaluate_pr_curve.py --model ./models/acf_detector.pkl --annotation_file ./data/wider_face_split/wider_face_val_bbx_gt.txt --image_dir ./data/WIDER_val/images/
```

6. Run the web app
```bash
streamlit run app/app.py
```
