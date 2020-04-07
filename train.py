import dataloader
import model
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from tqdm import tqdm

batchSize = 16
content_criterion = nn.MSELoss()
GeneratorDevice = torch.device("cuda:1")
DiscriminatorDevice = torch.device("cuda:0")
adversarial_criterion = nn.BCELoss()

train_dataloader = torch.utils.data.DataLoader(dataloader.train_dataset, batchSize, shuffle=True, num_workers=12)
valid_dataloader = torch.utils.data.DataLoader(dataloader.dev_dataset, 1, shuffle=False)

generator = model.Generator(16, dataloader.SCALING_FACTOR)
discriminator = model.Discriminator()
feature_extractor = model.FeatureExtractor()
writer = SummaryWriter()
generator.load_state_dict(torch.load('./checkpoint/generator_final.pth'))
discriminator.load_state_dict(torch.load('./checkpoint/discriminator_final.pth'))

generator = generator.to(GeneratorDevice)
discriminator = discriminator.to(DiscriminatorDevice)
feature_extractor = feature_extractor.to(DiscriminatorDevice)
low_res = torch.FloatTensor(batchSize, 3, dataloader.LOWRES, dataloader.LOWRES)
ones_const = torch.tensor(torch.ones(batchSize, 1), device=DiscriminatorDevice)



def train():
    optim_generator = optim.Adam(generator.parameters(), lr=0.0001)
    count = 0
    print("Generator PreTraining")

    for epoch in tqdm(range(2), desc ='pretraining'):
        for i, data in enumerate(train_dataloader):
            high_res_real, _ = data
            if high_res_real.shape[0] == batchSize:
                for j in range(batchSize):
                    low_res[j] = dataloader.scale(high_res_real[j])
                    high_res_real[j] = dataloader.normalize(high_res_real[j])

                high_res_real = high_res_real.to(GeneratorDevice)
                high_res_fake = generator(low_res.to(GeneratorDevice))

                generator.zero_grad()
                generator_content_loss = content_criterion(high_res_fake, high_res_real)
                generator_content_loss.backward()
                optim_generator.step()

                writer.add_scalar('data/pretrained_generator_content_loss', generator_content_loss, count)
                count += 1

    torch.save(generator.state_dict(), './checkpoint/generator.pth')

    generator_optimzer = optim.Adam(generator.parameters(), lr=0.00001)
    discriminator_optimzer = optim.Adam(discriminator.parameters(), lr=0.00001)
    count = 0
    print("SRGAN Training")

    for epoch in tqdm(range(500)):
        for i, data in enumerate(train_dataloader):
            high_res_real, _ = data
            if high_res_real.shape[0] == batchSize:
                for j in range(batchSize):
                    low_res[j] = dataloader.scale(high_res_real[j])
                    high_res_real[j] = dataloader.normalize(high_res_real[j])

                high_res_real = high_res_real.to(GeneratorDevice)
                high_res_fake = generator(low_res.to(GeneratorDevice))

                target_real = (torch.rand(batchSize, 1) * 0.5 + 0.7).to(DiscriminatorDevice)
                target_fake = (torch.rand(batchSize, 1) * 0.3).to(DiscriminatorDevice)
                high_res_fake = high_res_fake.to(DiscriminatorDevice)
                high_res_real = high_res_real.to(DiscriminatorDevice)

                # Train Discriminator#
                discriminator.zero_grad()
                discriminiator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + adversarial_criterion(discriminator(high_res_fake), target_fake)
                discriminiator_loss.backward(retain_graph=True)
                discriminator_optimzer.step()
                real_features = feature_extractor(high_res_real)
                fake_features = feature_extractor(high_res_fake)

                # Train Generator#
                generator.zero_grad()
                generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006 * content_criterion(fake_features, real_features)
                generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), ones_const)
                generator_total_loss = generator_content_loss + 0.001 * generator_adversarial_loss
                generator_total_loss.backward()
                generator_optimzer.step()

                writer.add_scalar('data/generator_content_loss,', generator_content_loss, count)
                writer.add_scalar('data/generator_adversarial_loss,', generator_adversarial_loss, count)
                writer.add_scalar('data/generator_total_loss,', generator_total_loss, count)
                writer.add_scalar('data/discriminator_loss,', discriminiator_loss, count)
                count += 1


if __name__ == "__main__":
    train()
    torch.save(generator.state_dict(), './checkpoint/generator_final.pth')
    torch.save(discriminator.state_dict(), './checkpoint/discriminator_final.pth')
    writer.close()