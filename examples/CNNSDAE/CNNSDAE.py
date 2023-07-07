import click
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from tensorboardX import SummaryWriter
from sklearn.cluster import KMeans
import uuid
import os,json,random
from PIL import Image


from sdaeCNN.sdae import StackedDenoisingAutoEncoder
import sdaeCNN.model as ae


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels



def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    #traIN/TEST/VAL
    train_val = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    train_val.sort()
    train_val_indices = dict((k, v) for v, k in enumerate(train_val))
    train_val_json_str = json.dumps(dict((val, key) for key, val in train_val_indices.items()), indent=4)


    for set in train_val:
        # train_images_path = []  # 存储训练集的所有图片路径
        # train_images_label = []  # 存储训练集图片对应索引信息
        # val_images_path = []  # 存储验证集的所有图片路径
        # val_images_label = []  # 存储验证集图片对应索引信息
        every_class_num = []  # 存储每个类别的样本总数
        if set == 'train':
            train_images_path = []  # 存储训练集的所有图片路径
            train_images_label = []  # 存储训练集图片对应索引信息
            roots = os.path.join(root, set)
            flower_class = [cla for cla in os.listdir(roots) if os.path.isdir(os.path.join(roots, cla))]
            # 排序，保证顺序一致
            flower_class.sort()
            # 生成类别名称以及对应的数字索引
            class_indices = dict((k, v) for v, k in enumerate(flower_class))
            json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
            with open('class_indices.json', 'w') as json_file:
                json_file.write(json_str)

            supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

            # for csl in flower_class:
            for csl in range(1):
                cla_path = roots
                vedio_number = []
                for cel in os.listdir(cla_path):
                    if os.path.isdir(os.path.join(cla_path, cel)):
                        vedio_number.append(os.path.join(cla_path, cel))
                video_number = [cel for cel in os.listdir(cla_path) if os.path.isdir(os.path.join(cla_path, cel))]
                video_number.sort()

                for cla in vedio_number:
                    # csl = cla
                    csl = os.listdir(cla)
                    for t in csl:
                        # 遍历获取supported支持的所有文件路径
                        video_roots = os.path.join(cla_path, cla,t)

                        # images = [os.path.join(video_roots, cla, i) for i in os.listdir(video_roots)
                        #           if os.path.splitext(i)[-1] in supported]
                        images = [os.path.join(video_roots, i) for i in os.listdir(video_roots)
                                  if os.path.splitext(i)[-1] in supported]
                        # 获取该类别对应的索引
                        # image_class = class_indices[csl]
                        image_class = class_indices[cla.split('/')[-1]]
                        # 记录该类别的样本数量
                        every_class_num.append(len(images))
                        for img_path in images:
                            train_images_path.append(img_path)
                            train_images_label.append(image_class)

            print("{} images for training.".format(len(train_images_path)))
        elif set == 'val':
            val_images_path = []  # 存储验证集的所有图片路径
            val_images_label = []  # 存储验证集图片对应索引信息
            roots = os.path.join(root, set)
            flower_class = [cla for cla in os.listdir(roots) if os.path.isdir(os.path.join(roots, cla))]
            # 排序，保证顺序一致
            flower_class.sort()
            # 生成类别名称以及对应的数字索引
            class_indices = dict((k, v) for v, k in enumerate(flower_class))
            json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
            with open('class_indices.json', 'w') as json_file:
                json_file.write(json_str)

            supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

            # for csl in flower_class:
            for csl in range(1):
                cla_path = roots
                vedio_number = []
                for cel in os.listdir(cla_path):
                    if os.path.isdir(os.path.join(cla_path, cel)):
                        vedio_number.append(os.path.join(cla_path, cel))
                video_number = [cel for cel in os.listdir(cla_path) if os.path.isdir(os.path.join(cla_path, cel))]
                video_number.sort()

                for cla in vedio_number:
                    # csl = cla
                    csl = os.listdir(cla)
                    for t in csl:
                        # 遍历获取supported支持的所有文件路径
                        video_roots = os.path.join(cla_path, cla, t)

                        # images = [os.path.join(video_roots, cla, i) for i in os.listdir(video_roots)
                        #           if os.path.splitext(i)[-1] in supported]
                        images = [os.path.join(video_roots, i) for i in os.listdir(video_roots)
                                  if os.path.splitext(i)[-1] in supported]
                        # 获取该类别对应的索引
                        # image_class = class_indices[csl]
                        image_class = class_indices[cla.split('/')[-1]]
                        # 记录该类别的样本数量
                        every_class_num.append(len(images))
                        for img_path in images:
                            val_images_path.append(img_path)
                            val_images_label.append(image_class)
            print("{} images for val.".format(len(val_images_path)))
        else:
            test_images_path = []  # 存储验证集的所有图片路径
            test_images_label = []  # 存储验证集图片对应索引信息
            roots = os.path.join(root, set)
            flower_class = [cla for cla in os.listdir(roots) if os.path.isdir(os.path.join(roots, cla))]
            # 排序，保证顺序一致
            flower_class.sort()
            # 生成类别名称以及对应的数字索引
            class_indices = dict((k, v) for v, k in enumerate(flower_class))
            json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
            with open('class_indices.json', 'w') as json_file:
                json_file.write(json_str)

            supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

            # for csl in flower_class:
            for csl in range(1):
                cla_path = roots
                vedio_number = []
                for cel in os.listdir(cla_path):
                    if os.path.isdir(os.path.join(cla_path, cel)):
                        vedio_number.append(os.path.join(cla_path, cel))
                video_number = [cel for cel in os.listdir(cla_path) if os.path.isdir(os.path.join(cla_path, cel))]
                video_number.sort()

                for cla in vedio_number:
                    # csl = cla
                    csl = os.listdir(cla)
                    for t in csl:
                        # 遍历获取supported支持的所有文件路径
                        video_roots = os.path.join(cla_path, cla, t)

                        # images = [os.path.join(video_roots, cla, i) for i in os.listdir(video_roots)
                        #           if os.path.splitext(i)[-1] in supported]
                        images = [os.path.join(video_roots, i) for i in os.listdir(video_roots)
                                  if os.path.splitext(i)[-1] in supported]
                        # 获取该类别对应的索引
                        # image_class = class_indices[csl]
                        image_class = class_indices[cla.split('/')[-1]]
                        # 记录该类别的样本数量
                        every_class_num.append(len(images))
                        for img_path in images:
                            test_images_path.append(img_path)
                            test_images_label.append(image_class)
            print("{} images for test.".format(len(test_images_path)))


            # print("{} images for training.".format(len(train_images_path)))

            # for cla in flower_class:
            #     cla_path = os.path.join(roots, cla)
            #     # 遍历获取supported支持的所有文件路径
            #     images = [os.path.join(roots, cla, i) for i in os.listdir(cla_path)
            #               if os.path.splitext(i)[-1] in supported]
            #     # 获取该类别对应的索引
            #     image_class = class_indices[cla]
            #     # 记录该类别的样本数量
            #     every_class_num.append(len(images))
            #     for img_path in images:
            #         val_images_path.append(img_path)
            #         val_images_label.append(image_class)




    # 遍历文件夹，一个文件夹对应一个类别
    # flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # # 排序，保证顺序一致
    # flower_class.sort()
    # # 生成类别名称以及对应的数字索引
    # class_indices = dict((k, v) for v, k in enumerate(flower_class))
    # json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    # with open('class_indices.json', 'w') as json_file:
    #     json_file.write(json_str)
    #
    # train_images_path = []  # 存储训练集的所有图片路径
    # train_images_label = []  # 存储训练集图片对应索引信息
    # val_images_path = []  # 存储验证集的所有图片路径
    # val_images_label = []  # 存储验证集图片对应索引信息
    # every_class_num = []  # 存储每个类别的样本总数
    # supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # # 遍历每个文件夹下的文件
    # for cla in flower_class:
    #     cla_path = os.path.join(root, cla)
    #     # 遍历获取supported支持的所有文件路径
    #     images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
    #               if os.path.splitext(i)[-1] in supported]
    #     # 获取该类别对应的索引
    #     image_class = class_indices[cla]
    #     # 记录该类别的样本数量
    #     every_class_num.append(len(images))
    #     # 按比例随机采样验证样本
    #     val_path = random.sample(images, k=int(len(images) * val_rate))
    #
    #     for img_path in images:
    #         if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
    #             val_images_path.append(img_path)
    #             val_images_label.append(image_class)
    #         else:  # 否则存入训练集
    #             train_images_path.append(img_path)
    #             train_images_label.append(image_class)
    #
    # print("{} images were found in the dataset.".format(sum(every_class_num)))
    # print("{} images for training.".format(len(train_images_path)))
    # print("{} images for validation.".format(len(val_images_path)))

    # plot_image = False
    # if plot_image:
    #     # 绘制每种类别个数柱状图
    #     plt.bar(range(len(flower_class)), every_class_num, align='center')
    #     # 将横坐标0,1,2,3,4替换为相应的类别名称
    #     plt.xticks(range(len(flower_class)), flower_class)
    #     # 在柱状图上添加数值标签
    #     for i, v in enumerate(every_class_num):
    #         plt.text(x=i, y=v + 5, s=str(v), ha='center')
    #     # 设置x坐标
    #     plt.xlabel('image class')
    #     # 设置y坐标
    #     plt.ylabel('number of images')
    #     # 设置柱状图的标题
    #     plt.title('flower class distribution')
    #     plt.show()
    '''修改3'''
    # if len(train_images_path)/32 !=0:
    #     ttt = int((len(train_images_path)/32) //2) * 2
    #     train_images_path = train_images_path[:ttt*32]
    #     train_images_label = train_images_label[:ttt*32]
    # if len(val_images_path)/32 !=0:
    #     ttt = int((len(val_images_path)/32) //2) * 2
    #     val_images_path = val_images_path[:ttt*32]
    #     val_images_label = val_images_label[:ttt*32]
    '''修改3'''
    return train_images_path, train_images_label, val_images_path, val_images_label,test_images_path, test_images_label
@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=True
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=16
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=800,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=200,
)
@click.option(
    "--testing-mode",
    help="whether to run in testing mode (default False).",
    type=bool,
    default=False,
)
def main(cuda, batch_size, pretrain_epochs, finetune_epochs, testing_mode):
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope

    pre_train_unLock = False #控制是否进行预训练
    train_unLock = True #控制是否进行分类训练

    checkpoint_path = f'pretrainModel.pth'

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )

    train_images_path, train_images_label, val_images_path, val_images_label,test_images_path, test_images_label = read_split_data("/home/wangkai/Dataset/datasets")

    img_size = {"s": [300, 384],  # train_size, val_size
                "m": [384, 480],
                "l": [384, 480],
                "xs": [256, 256]}

    num_model = "xs"

    data_transform = {
        "train": transforms.Compose([transforms.Resize(img_size[num_model]),
                                     transforms.ToTensor(),

                                     ]),
        "val": transforms.Compose([transforms.Resize(img_size[num_model]),
                                   transforms.ToTensor(),
                                   ])}

    # 实例化训练数据集
    ds_train = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    ds_val = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    #

    # evaluation dataset
    autoencoder = StackedDenoisingAutoEncoder(
        [[3,32,16,8],[8,4,4,1]],final_activation=None
    )
    print(autoencoder)
    if cuda:
        autoencoder.cuda()

    if(pre_train_unLock):
        print("Pretraining stage.")
        ae.pretrain(
            ds_train,
            autoencoder,
            cuda=cuda,
            validation=ds_val,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            optimizer=lambda model: SGD(model.parameters(), lr=0.001, momentum=0.9),
            scheduler=lambda x: StepLR(x, 100, gamma=0.1),
            corruption=0.2,
            num_workers=6,
            update_freq=1,
        )
        torch.save(autoencoder.state_dict(), checkpoint_path)
    if(train_unLock):
        # autoencoder.load_state_dict(torch.load(checkpoint_path))
        print("Training stage.")

        autoencoder = autoencoder.encoder
        ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
        ae.finetune(
            ds_train,
            autoencoder,
            cuda=cuda,
            validation=ds_val,
            epochs=finetune_epochs,
            batch_size=batch_size,
            optimizer=ae_optimizer,
            scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
            # corruption=0.2,
            num_workers=6,
            update_callback=training_callback,
        )



if __name__ == "__main__":
    main()
