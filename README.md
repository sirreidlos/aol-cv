# ACF impl

0. Clone repository and install dependencies 
```bash
git clone git@github.com:sirreidlos/aol-cv.git
cd aol-cv
# I recommend making a venv beforehand
pip install -r requirements.txt # or uv sync
```

1. Download the MUCT dataset
[MUCT Face Database](http://www.milbo.org/muct/)

Download all the archived files and then unarchive them 

```bash
mkdir data
tar -xvf muct-a-jpg-v1.tar.gz -C data/
tar -xvf muct-b-jpg-v1.tar.gz -C data/
tar -xvf muct-c-jpg-v1.tar.gz -C data/
tar -xvf muct-d-jpg-v1.tar.gz -C data/
tar -xvf muct-e-jpg-v1.tar.gz -C data/
tar -xvf muct-landmarks-v1.tar.gz -C data/
```

Ensure that the folder structure looks like this:
```
    ┌── muct76-opencv.csv     
  ┌─┴ muct-landmarks
  │ ┌── *.jpg
  ├─┴ jpg
┌─┴ data                  
```

2. Run training

Example:
```bash
python train.py --dataset muct --annotation_file data/muct-landmarks/muct76-opencv.csv --image_dir data/jpg/ --output_model models/muct_mlp.pkl --selection_metric f1
python train.py --dataset muct --annotation_file data/muct-landmarks/muct76-opencv.csv --image_dir data/jpg/ --output_model models/muct_ada.pkl --selection_metric f1 --model ada
python train.py --dataset muct --annotation_file data/muct-landmarks/muct76-opencv.csv --image_dir data/jpg/ --output_model models/muct_gbm.pkl --selection_metric f1 --model gbm
```

3. Run inference

Example:
```bash
python inference.py --model ./models/muct_mlp.pkl --image data/jpg/i000re-fn.jpg --output ./output1.png --n_per_oct 8 --n_oct_up 1 --min_ds 256 256 --stride 16
```

4. Run evaluation
Example:
```bash
python evaluate.py --dataset muct --annotation_file ./data/muct-landmarks/muct76-opencv.csv --image_dir ./data/jpg/ --n_per_oct 8 --n_oct_up 1 --max_ds 256 256 --stride 16 --batch_size 4096 --model ./models/muct_mlp.pkl
```

5. Run evaluation to compute PR curve and AP
```bash
python evaluate_pr_curve.py --dataset muct --model ./models/muct_ada.pkl --annotation_file ./data/muct-landmarks/muct76-opencv.csv --image_dir ./data/jpg/ --batch_size 2048 --n_per_oct 8 --n_oct_up 1 --max_ds 256 256 --stride 16
```

6. Run the web app
```bash
streamlit run app/app.py
```
