from dataset.imagenet_dataset import TinyImageNet
data_dir = './dataset/imagenet/tiny-imagenet-200'
dataset_train = TinyImageNet(data_dir, train=True)
dataset_val = TinyImageNet(data_dir, train=False)
print(dataset_train[0])
