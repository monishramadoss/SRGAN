import data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
dev_folder = data.DEVDATA_FOLDER
train_folder = data.TRAINDATA_FOLDER
SCALING_FACTOR = 2
LOWRES = 112

transform = transforms.Compose([transforms.RandomCrop(LOWRES*SCALING_FACTOR),
                                transforms.ToTensor()])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

scale = transforms.Compose([transforms.ToPILImage(), transforms.Resize(LOWRES),
                            transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_dataset = datasets.ImageFolder(root='./data/train', transform=transform)
dev_dataset = datasets.ImageFolder(root='./data/dev', transform=transform)
