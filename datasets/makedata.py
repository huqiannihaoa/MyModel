# 该文件我也放在VOC2007的文件夹运行的，放在其他目录下运行应该也没什么问题，不过路径需要使用绝对路径，否则会报错

import shutil
import os

file_List = ["test"]
for file in file_List:
    if not os.path.exists('./datasets/VOC/test2007/images/%s' % file):
        os.makedirs('./datasets/VOC/test2007/images/%s' % file)
    if not os.path.exists('./datasets/VOC/test2007/labels/%s' % file):
        os.makedirs('./datasets/VOC/test2007/labels/%s' % file)
    print(os.path.exists('./%s.txt' % file))
    f = open('./%s.txt' % file, 'r')
    lines = f.readlines()
    for line in lines:
        print(line)
        line = "/".join(line.split('/')[-6:]).strip()
        shutil.copy(line, "./datasets/VOC/test2007/images/%s" % file)
        line = line.replace('JPEGImages', 'labels')
        line = line.replace('jpg', 'txt')
        shutil.copy(line, "./datasets/VOC/test2007/labels/%s/" % file)

# 该文件我也放在VOC2007的文件夹运行的，放在其他目录下运行应该也没什么问题，不过路径需要使用绝对路径，否则会报错

# import shutil
# import os
#
# file_List = ["train", "val"]
# for file in file_List:
#     if not os.path.exists('./datasets/VOC2012/images/%s' % file):
#         os.makedirs('./datasets/VOC2012/images/%s' % file)
#     if not os.path.exists('./datasets/VOC2012/labels/%s' % file):
#         os.makedirs('./datasets/VOC2012/labels/%s' % file)
#     print(os.path.exists('./%s.txt' % file))
#     f = open('./%s.txt' % file, 'r')
#     lines = f.readlines()
#     for line in lines:
#         print(line)
#         line = "/".join(line.split('/')[-6:]).strip()
#         #print(line)
#         shutil.copy(line, "./datasets/VOC2012/images/%s" % file)
#         line = line.replace('JPEGImages', 'labels')
#         line = line.replace('jpg', 'txt')
#         shutil.copy(line, "./datasets/VOC2012/labels/%s/" % file)



# file_List = ["train","val"]
# for file in file_List:
#     if not os.path.exists('./datasets/VOC/VOC2007/images/%s' % file):
#         os.makedirs('./datasets/VOC/VOC2007/images/%s' % file)
#     if not os.path.exists('./datasets/VOC/VOC2007/labels/%s' % file):
#         os.makedirs('./datasets/VOC/VOC2007/labels/%s' % file)
#     print(os.path.exists('./%s.txt' % file))
#     f = open('./%s.txt' % file, 'r')
#     lines = f.readlines()
#     for line in lines:
#         print(line)
#         line = "/".join(line.split('/')[-5:]).strip()
#         shutil.copy(line, "./datasets/VOC/VOC2007/images/%s" % file)
#         line = line.replace('JPEGImages', 'labels')
#         line = line.replace('jpg', 'txt')
#         shutil.copy(line, "./datasets/VOC/VOC2007/labels/%s/" % file)

# file_List = ["train","val"]
# for file in file_List:
#     if not os.path.exists('./datasets/VOC/test/images/%s' % file):
#         os.makedirs('./datasets/VOC/VOC2007/images/%s' % file)
#     if not os.path.exists('./datasets/VOC/test/labels/%s' % file):
#         os.makedirs('./datasets/VOC/test/labels/%s' % file)
#     print(os.path.exists('./%s.txt' % file))
#     f = open('./%s.txt' % file, 'r')
#     lines = f.readlines()
#     for line in lines:
#         print(line)
#         line = "/".join(line.split('/')[-5:]).strip()
#         shutil.copy(line, "./datasets/VOC/test/images/%s" % file)
#         line = line.replace('JPEGImages', 'labels')
#         line = line.replace('jpg', 'txt')
#         shutil.copy(line, "./datasets/VOC/test/labels/%s/" % file)