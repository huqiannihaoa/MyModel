import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

pwd = os.getcwd()

# names = ['yolov5+Conv-SPD+DAFPN(EIoU)','yolov5+Conv-SPD+DAFPN(CIoU)','yolov5+Conv-SPD+DAFPN(ECIoU)']
names = ['yolov5(ECIoU)','yolov5(EIoU)','yolov5+Conv-SPD+DAFPN(EIoU)','yolov5+Conv-SPD+DAFPN(CIoU)','yolov5+Conv-SPD+DAFPN(ECIoU)']
# names = ['mymodel','mymodel02-eciou']
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
for i in names:
    data = pd.read_csv(f'runs/train/xr/widerperson/{i}/results.csv')
    plt.plot(data['   metrics/precision'], label=i)
plt.xlabel('epoch')
plt.title('precision')
plt.legend()

plt.subplot(2, 2, 2)
for i in names:
    data = pd.read_csv(f'runs/train/xr/widerperson/{i}/results.csv')
    plt.plot(data['      metrics/recall'], label=i)
plt.xlabel('epoch')
plt.title('recall')
plt.legend()

plt.subplot(2, 2, 3)
for i in names:
    data = pd.read_csv(f'runs/train/xr/widerperson/{i}/results.csv')
    plt.plot(data['     metrics/mAP_0.5'], label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5')
plt.legend()

plt.subplot(2, 2, 4)
for i in names:
    data = pd.read_csv(f'runs/train/xr/widerperson/{i}/results.csv')
    plt.plot(data['metrics/mAP_0.5:0.95'], label=i)
plt.xlabel('epoch')
plt.title('   mAP_0.5:0.95')
plt.legend()

plt.tight_layout()
plt.savefig('metrice_curve.png')
print(f'metrice_curve.png save in {pwd}/metrice_curve.png')

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
for i in names:
    data = pd.read_csv(f'runs/train/xr/widerperson/{i}/results.csv')
    plt.plot(data['      train/box_loss'], label=i)
plt.xlabel('epoch')
plt.title('train/box_loss')
plt.legend()

plt.subplot(2, 2, 2)
for i in names:
    data = pd.read_csv(f'runs/train/xr/widerperson/{i}/results.csv')
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
    data = pd.read_csv(f'runs/train/xr/widerperson/{i}/results.csv')
    plt.plot(data['        val/box_loss'], label=i)
plt.xlabel('epoch')
plt.title('val/box_loss')
plt.legend()

plt.subplot(2, 2, 4)
for i in names:
    data = pd.read_csv(f'runs/train/xr/widerperson/{i}/results.csv')
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