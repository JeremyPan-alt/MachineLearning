import warnings
from ultralytics import YOLO
warnings.filterwarnings('ignore')

# 指定显卡和多卡训练问题 统一都在<YOLOV8V10配置文件.md>下方常见错误和解决方案。
# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolov11-infrared-SHSA.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    path = 'ultralytics/cfg/datasets/solar-deffects-Dataset.yaml'
    model.train(data=path,
                cache=False,
                imgsz=640,
                epochs=200,
                lr0=0.1,
                batch=64,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD',  # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='exp',
                )
