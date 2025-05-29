这是DATA620004_homework2_task2的作业仓库

运行方式：

1.安装环境依赖，其中mmcv版本为2.1.0，mmdet版本为3.3.0。

2.运行download_data.py下载并解压VOC数据集。

3.运行python data/VOCdevkit/voc2coco.py将VOC转换为coco数据集，运行data/VOCdevkit/segmentation.py脚本添加分割标注信息，脚本的路径需要根据实际存放路径修改。

4.运行python tools/train.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py训练mask-rcnn网络。

5.运行python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py训练sparse-rcnn网络。

以上操作会自动生成work_dirs文件夹并记录tensorboard日志和每个epoch的模型，也可直接通过https://pan.baidu.com/s/1gmjetgNOsTre6pEuo22b4w?pwd=jk6a 下载训练好的模型，将work_dirs文件架解压在根目录下。

6.数据可视化：修改tools/visualize_with_segmentation.py脚本中的图片路径信息，运行python tools/visualize_with_segmentation.py脚本，可视化输出保存在检测图片同级文件夹下。