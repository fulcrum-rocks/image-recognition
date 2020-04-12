# Train YOLO3 Eager

How to Create Neural Network for Recognizing Masks

To start need:

1. Special Yolov3
2. Dataset
3. Imgs
4. yolov3.weights
5. Video

// ==================== train ======================

python src/train_eager.py -c configs/mask_500.json

// ==================== read picture ======================

python src/pred.py -c configs/mask_500.json -i imgs/1.jpg

// ===================== Video =====================

python video.py -c configs/mask_500.json -i videoplayback.mp4

// ==================== test / benchmark ======================

python src/eval.py -c configs/mask_500.json

best result => {'fscore': 0.8120770432066632, 'precision': 0.8676307007786429, 'recall': 0.7632093933463796}

Yolo3 instalation:

$ activate yolo3 # in linux "source activate yolo3"
(yolo3) $ pip install -r requirements.txt
(yolo3) \$ pip install -e .

Dataset structure:

dataset/name/train/images|annotations

Link to default yolo.weights

https://pjreddie.com/media/files/yolov3.weights
