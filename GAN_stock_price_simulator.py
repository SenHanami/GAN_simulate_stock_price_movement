import torch
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import config
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
from matplotlib import pyplot as plt
import os


class sMIndexOpen_fake_dataset(Dataset):

    def __init__(self, model, fake_only=True):
        self.model = model
        self.batch_size = config.BATCH_SIZE

    def __getitem__(self, idx):
        input = torch.tensor(np.random.normal(0, 1, 100)).unsqueeze(0)
        return self.model(input).squeeze(0), 0
        # random.uniform(0.1,0.3)

    def __len__(self):
        return config.FAKE_SAMPLE_NUM


class sMIndexOpen_real_dataset(Dataset):
    def __init__(self):
        with open('./data/rate_data', 'rb')as f:
            rate_data = pickle.load(f)
            self.rate_data = rate_data
            print('rate_data: ', rate_data)

    def __getitem__(self, idx):
        return self.rate_data[idx:idx + config.SAMPLE_LENGTH], 1



    def __len__(self):
        return self.rate_data.shape[0] - config.SAMPLE_LENGTH + 1


class Noise_generator(DataLoader):
    def __init__(self):
        pass

    def __getitem__(self, idx):
        return np.random.normal(0, 1, 100), 0

    def __len__(self):
        return config.FAKE_SAMPLE_NUM


class Genetator(nn.Module):
    def __init__(self):
        super(Genetator, self).__init__()
        self.fc1 = nn.Linear(100, 8)
        self.BatchNorm1d_1 =nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 64)
        self.BatchNorm1d_2 =nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 256)
        self.BatchNorm1d_3 =nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 512)
        self.BatchNorm1d_4 =nn.BatchNorm1d(512)
        self.fc5 = nn.Linear(512, config.SAMPLE_LENGTH)
        self.BatchNorm1d_5 =nn.BatchNorm1d(config.SAMPLE_LENGTH)
        self.fc6 = nn.Linear(256, 512)
        self.Dropout_1 = nn.Dropout(0.2)
        self.Dropout_2 = nn.Dropout(0.3)
        self.Dropout_3 = nn.Dropout(0.2)
        self.Dropout_4 = nn.Dropout(0.3)

    def forward(self, input):
        x = self.fc1(input)
        # x =self.BatchNorm1d_1(x)
        x = torch.relu(x)

        # x =self.Dropout_1(x)
        x = self.fc2(x)
        # x =self.BatchNorm1d_2(x)
        x =torch.relu(x)

        # x =self.Dropout_3(x)
        x =self.fc3(x)
        # x =self.BatchNorm1d_3(x)
        x = torch.sigmoid(x)

        # x =self.Dropout_4(x)
        x = self.fc4(x)
        # x = self.BatchNorm1d_4(x)
        x = torch.tanh(x)

        x = self.fc5(x)
        output = torch.tanh(x)


        return output


class Disciminator(nn.Module):
    def __init__(self):
        super(Disciminator, self).__init__()
        self.fc1 = nn.Linear(config.SAMPLE_LENGTH, 512)
        self.BatchNorm1d_1 =nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.BatchNorm1d_2 =nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 1)
        self.Dropout_1 = nn.Dropout(0.2)
        self.Dropout_2 = nn.Dropout(0.3)

    def forward(self, input):
        x = self.fc1(input)
        # x =self.BatchNorm1d_1(x)
        x = torch.relu(x)

        # x =self.Dropout_1(x)
        x = self.fc2(x)
        # x =self.BatchNorm1d_2(x)
        x = torch.relu(x)

        # x =self.Dropout_2(x)
        x = self.fc3(x)

        output = F.relu(x)
        output = torch.sigmoid(x)

        return output


def train(generator, generator_optimizer, discriminator, discriminator_optimizer, sMIndexOpen_real_dataset,sMIndexOpen_fake_dataset, noise_generator):
    # train disctiminator
    criterion = nn.BCELoss()
    # generator.train()
    for idx0, (real_input, real_target) in enumerate(DataLoader(sMIndexOpen_real_dataset, batch_size=100, drop_last=True, shuffle=True)):
        for idx1, (noise, place_holder) in enumerate(DataLoader(noise_generator, batch_size=100)):
            discriminator_optimizer.zero_grad()

            fake_input = generator(noise)
            fake_target = torch.zeros(fake_input.shape[0])

            total_input = torch.cat([real_input, fake_input], dim=0)
            total_target = torch.cat([real_target, fake_target], dim=0)

            output = discriminator(total_input)
            # loss0 = F.nll_loss(output, total_target.long())
            loss0 =criterion(output,total_target.double())
            # pred = output.max(dim=-1)[-1]
            # acc = 100. * pred.eq(total_target.data).numpy().mean()

            loss0.backward()
            discriminator_optimizer.step()
            break
        if idx0%20 ==0:
            print('discriminator loss:',loss0)

        # if idx % 20 == 0:
        #     print('discriminator loss: ', loss0)
        #     print('区分的准确率: ', acc)

    #train generator
    # generator.train()
    for idx3, (real_input, real_target) in enumerate(DataLoader(sMIndexOpen_real_dataset, batch_size=100, drop_last=True, shuffle=True)):
        for idx1, (noise, place_holder) in enumerate(DataLoader(noise_generator, batch_size=100)):
            generator_optimizer.zero_grad()

            fake_input = generator(noise)
            fake_target = torch.zeros(fake_input.shape[0])



            total_input = torch.cat([real_input, fake_input], dim=0)
            total_target = torch.cat([real_target, fake_target], dim=0)

            output = discriminator(total_input)
            # loss1 = -F.nll_loss(output, total_target.long())
            loss1 = -criterion(output,total_target.double())

            pred = output.max(dim=-1)[-1]
            acc = 100. * pred.eq(total_target.data).numpy().mean()

            loss1.backward()
            generator_optimizer.step()
            break

        if idx3 % 20 == 0:
            print('generator loss1: ', loss1)
        #     print('区分的准确率: ', acc)
    # # train genetator2
    # for idx, (noise, place_holder) in enumerate(DataLoader(noise_generator, batch_size=24)):
    #     generator.zero_grad()
    #     fake_input = generator(noise)
    #     fake_target = torch.zeros(fake_input.shape[0])
    #     output = discriminator(fake_input)
    #     loss1 = -F.nll_loss(output, fake_target.long())
    #     if idx % 20 == 0:
    #         print('generator_loss: ', loss1)
    #     loss1.backward()
    #     generator_optimizer.step()

    # 画图
    # generator.eval()
    noise =torch.tensor(np.random.normal(0, 1, 100), dtype=torch.float64).unsqueeze(0)
    fake_data = generator(noise).detach().numpy().ravel()
    y = (fake_data + 1).cumprod()
    x = np.linspace(1, fake_data.shape[0], fake_data.shape[0])
    # print('收益率： ',fake_data)
    # print('y:', y)
    # print(x)
    # print(fake_data)
    plt.figure(figsize=(10, 5))
    plt.plot(y)
    plt.show()

    # torch.save(discriminator.state_dict(), './models/discriminator_model.pkl')
    # torch.save(discriminator_optimizer.state_dict(), './optimizers/discriminator_optimizer.pkl')
    # torch.save(generator.state_dict(), './models/generator_model.pkl')
    # torch.save(generator_optimizer.state_dict(), './optimizers/generator_optimizer.pkl')


if __name__ == '__main__':
    generator = Genetator().double()
    discriminator = Disciminator().double()
    generator_optimizer = Adam(generator.parameters(), lr=1e-5)
    discriminator_optimizer = Adam(discriminator.parameters(), lr=1e-5)

    noise_generator = Noise_generator()
    sMIndexOpen_real_dataset = sMIndexOpen_real_dataset()
    # sMIndexOpen_fake_dataset = sMIndexOpen_fake_dataset(generator)

    for i in range(100):
        train(generator, generator_optimizer, discriminator, discriminator_optimizer, sMIndexOpen_real_dataset,
              sMIndexOpen_fake_dataset, noise_generator)

    # #加载模型
    # if os.path.exists('./models/generator_model.pkl'):
    #     generator.load_state_dict(torch.load('./models/generator_model.pkl'))

    # 画图
    # fake_data  =generator(torch.tensor(np.random.normal(0,1,100),dtype=torch.float64)).detach().numpy().ravel()
    # x = np.linspace(1,fake_data.shape[0],fake_data.shape[0])
    # plt.figure(figsize=(50,30))
    # plt.scatter(x,fake_data)
    # plt.show()
