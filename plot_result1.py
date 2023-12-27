import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

pwd = os.getcwd()

names = ['YOLOv5m','YOLOv5s','YOLOv5n','YOLOv5s6','YOLOv5m6','YOLOv5-Transformer','YOLOv5_Conv-SPD_DAFPN','YOLOv6','YOLOv7','YOLOv8n','YOLOv8-AFPN','YOLOv8-DCNv2','YOLOv8-SCConv']
# names = ['120','120-eciou']

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
for i in names:
    data = pd.read_csv(f'runs/train/wpexp/{i}/results.csv')
    plt.plot(data['   metrics/precision'], label=i)
plt.xlabel('epoch')
plt.title('precision')
plt.legend()

plt.subplot(2, 2, 2)
for i in names:
    data = pd.read_csv(f'runs/train/wpexp/{i}/results.csv')
    plt.plot(data['      metrics/recall'], label=i)
plt.xlabel('epoch')
plt.title('recall')
plt.legend()

plt.subplot(2, 2, 3)
for i in names:
    data = pd.read_csv(f'runs/train/wpexp/{i}/results.csv')
    plt.plot(data['     metrics/mAP_0.5'], label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5')
plt.legend()

plt.subplot(2, 2, 4)
for i in names:
    data = pd.read_csv(f'runs/train/wpexp/{i}/results.csv')
    plt.plot(data['metrics/mAP_0.5:0.95'], label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5:0.95')
plt.legend()

plt.tight_layout()
plt.savefig('metrice_curve.png')
print(f'metrice_curve.png save in {pwd}/metrice_curve.png')

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
for i in names:
    data = pd.read_csv(f'runs/train/wpexp/{i}/results.csv')
    plt.plot(data['      train/box_loss'], label=i)
plt.xlabel('epoch')
plt.title('train/box_loss')
plt.legend()

plt.subplot(2, 2, 2)
for i in names:
    data = pd.read_csv(f'runs/train/wpexp/{i}/results.csv')
    plt.plot(data['      train/obj_loss'], label=i)
plt.xlabel('epoch')
plt.title('train/obj_loss')
plt.legend()

# plt.subplot(2, 3, 3)
# for i in names:
#     data = pd.read_csv(f'runs/train/humanexp/{i}/results.csv')
#     plt.plot(data['      train/cls_loss'], label=i)
# plt.xlabel('epoch')
# plt.title('train/cls_loss')
# plt.legend()

plt.subplot(2, 2, 3)
for i in names:
    data = pd.read_csv(f'runs/train/wpexp/{i}/results.csv')
    plt.plot(data['        val/box_loss'], label=i)
plt.xlabel('epoch')
plt.title('val/box_loss')
plt.legend()

plt.subplot(2, 2, 4)
for i in names:
    data = pd.read_csv(f'runs/train/wpexp/{i}/results.csv')
    plt.plot(data['        val/obj_loss'], label=i)
plt.xlabel('epoch')
plt.title('val/obj_loss')
plt.legend()

# plt.subplot(2, 3, 6)
# for i in names:
#     data = pd.read_csv(f'runs/train/humanexp/{i}/results.csv')
#     plt.plot(data['        val/cls_loss'], label=i)
# plt.xlabel('epoch')
# plt.title('val/cls_loss')
# plt.legend()

plt.tight_layout()
plt.savefig('loss_curve.png')
print(f'loss_curve.png save in {pwd}/loss_curve.png')