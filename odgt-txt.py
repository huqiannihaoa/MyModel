import os
import json
import argparse
import time

from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--classpath', type=str, default='class.txt', help='类别txt文件路径')
parser.add_argument('--inpath', type=str, default='data', help='输入文件/文件夹路径')
parser.add_argument('--outpath', type=str, default='out', help='输出文件夹路径')
parser.add_argument('--imgpath', type=str, default='img', help='图片文件夹路径')
parser.add_argument('--auto', type=str, default='no', help='是否自动适应并添加类别')

args = parser.parse_args()

classpath = args.classpath
inpath = args.inpath
outpath = args.outpath
imgpath = args.imgpath
auto = args.auto
if args.auto is not "yes":
    if args.auto is not "no":
        auto = "no"
        print("please check your parm, the --auto can only input with yes or no")
else:
    auto = "yes"

# 加载类别映射
# f_ids = open(classpath, 'r+')
# classnames = f_ids.read().splitlines()
categories = {"person": 0}  # 设置类别和id映射 名字在列表中的下标即为id


# 加载数据 根据入参inpath是否文件决定是否需要遍历
def load_func(inpath):
    if os.path.isdir(inpath):
        files = os.listdir(inpath)
    else:
        files = [inpath]
    records = []
    for file in files:
        assert os.path.exists(file)
        with open(file, 'r') as fid:
            lines = fid.readlines()
            records = records + [json.loads(line.strip('\n')) for line in lines]
    return records


# 计时函数
def timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {end - start} seconds')
        return result

    return wrapper


'''
    文件内指定目录分别需要指定pwd、inpath、object_dir三个目录，不用时注掉
    使用时目录文件树指定如下：
    inpath可以是单个或一个文件夹，是文件夹时在其中遍历所有文件
    object_dir中必须含有对应的图片路径images，
    就是说只有label目录可以在没有情况下自动创建，其他路径必须是要有

        pwd
            |inpath
            |object_dir
                |images
                |labels

'''
# 指定相对路径
pwd = "./human/"
# 待转换文件
inpath = "annotation_train.odgt"
inpath = os.path.join(pwd, inpath)
# 指定数据集
object_dir = "train"
object_dir = os.path.join(pwd, object_dir)
# 图片目录
imgpath = os.path.join(object_dir, "images")
# 标签输出目录
outpath = os.path.join(object_dir, "labels")

# 1. 处理输出路径
# 1.1 统一处理为绝对路径
if outpath.startswith(('\\', '/')):
    outpath = os.path.join(os.getcwd(), outpath[1:])

# 1.2. 路径不存在时创建
if not os.path.exists(outpath):
    os.mkdir(outpath)


def transform(start, length) -> float:
    half = length / 2
    if half > start:
        if start < half * 0.75:
            return 0
        return start
    elif start + half > 1:
        out_length = (1 - start) * 2 - 0.0001
        if out_length < length * 0.75:
            return 0
        return out_length
    return length


# 2. 执行转化
@timing
def convert():
    # 3. 加载文件
    records = load_func(inpath)
    import os

    for record in records:
        # 3.1 读取图片id
        id = record['ID']
        # 3.2 拼接文件路径
        txt_name = os.path.join(outpath, id + '.txt')
        img_path = os.path.join(imgpath, id + '.jpg')

        if not os.path.exists(img_path):
            continue  # 跳到下一张图片的处理

        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except (FileNotFoundError, OSError):
            os.remove(img_path)  # 删除损坏的图片
            continue  # 跳到下一张图片的处理

    # for record in records:
    #     # 3.1 读取图片id
    #     id = record['ID']
    #     # 3.2 拼接文件路径
    #     txt_name = os.path.join(outpath, id + '.txt')
    #     img_path = os.path.join(imgpath, id + '.jpg')
    #
    #     try:
    #         im = Image.open(img_path)
    #         img_w, img_h = im.size
    #         im.close()
    #     except FileNotFoundError:
    #         # print(f'WARNING: Image {id} not found. Skipping.')
    #         continue  # 跳到下一张图片的处理

        f = open(txt_name, 'w')  # 打开文件
        gtboxes = record['gtboxes']
        for box in gtboxes:
            category = box['tag']
            if category not in categories:
                if auto is "no":
                    continue
                elif auto is "yes":
                    categories[category] = len(categories)
                    f_ids.write(f'\n {category}')
            category_id = categories[category]

            x, y, w, h = box['fbox']
            # 计算中心坐标
            x = (x + w / 2) / img_w
            y = (y + h / 2) / img_h

            # 归一化宽高
            w = w / img_w
            h = h / img_h
            w = transform(x, w)
            if w == 0:
                continue
            h = transform(y, h)
            if h == 0:
                continue

            # print(category_id, x, y, w, h)
            f.write(f'{category_id} {x:.5f} {y:.5f} {w:.5f} {h:.5f}\n')
        f.close()
    print("转换完成！")


convert()
# f_ids.close()