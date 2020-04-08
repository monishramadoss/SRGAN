import dataloader
import model
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

writer = SummaryWriter()
valid_criterion = nn.MSELoss()
generator = model.Generator(16, dataloader.SCALING_FACTOR)
generator.load_state_dict(torch.load("./checkpoint/generator_final.pth"))
generator.eval()

valid_dataloader = torch.utils.data.DataLoader(dataloader.dev_dataset, 1)
unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])
interpolate = transforms.Compose([transforms.ToPILImage(), transforms.Resize(dataloader.LOWRES*dataloader.SCALING_FACTOR), transforms.ToTensor()])

def eval():
    count = 0
    generator.cuda()
    low_res = torch.zeros((1, 3, dataloader.LOWRES, dataloader.LOWRES), device=torch.device('cuda:0'))
    for i, data in enumerate(valid_dataloader):
        high_res_real, _ = data

        low_res[0] = dataloader.scale(high_res_real[0])
        high_res_fake = generator(low_res)
        valid_loss = valid_criterion(high_res_fake.cpu(), high_res_real)
        writer.add_scalar("./data/valid_loss", valid_loss, i)
        for i in range(1):
            writer.add_image('validation_image_real', high_res_real[i].cpu(), count)
            output = unnormalize(high_res_fake[i].cpu()).clamp(min=0, max=1)
            writer.add_image('validation_image', output, count)
            writer.add_image('validation_image_inter', interpolate(unnormalize(low_res[i].cpu())), count)
            count += 1



if __name__ == '__main__':
    eval()
