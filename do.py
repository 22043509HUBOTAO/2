import make_data_set
import ImageTransform
import torch
import torch.utils
import torch.utils.data
import torchvision
from tqdm import tqdm

size = 224
# mean = (0.5, 0.5, 0.5)
std = (0.220, 0.220, 0.220)
mean = (0, 0, 0)
k=6

train_list = make_data_set.make_data_path_list("train")
print("train size:{0}".format(len(train_list)))
val_list = make_data_set.make_data_path_list("val")
print("val size:{0}".format(len(val_list)))

train_dataset = \
    make_data_set.CancerDataset(file_list=train_list,
                                transform=ImageTransform.ImageTransform(size, mean, std),
                                phase='train')

val_dataset = \
    make_data_set.CancerDataset(file_list=val_list,
                                transform=ImageTransform.ImageTransform(size, mean, std),
                                phase='val')

batch_size = 32

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)

val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

batch_iterator = iter(dataloaders_dict["train"])

# inputs, labels = next(
#     batch_iterator
# )
# print(inputs.size())
# print(labels)

use_pretrained = True

net = torchvision.models.vgg16(pretrained=False)

print(net)

net.classifier[6] = torch.nn.Linear(in_features=4096, out_features=2)

net.train()

print("net setting successful")

criterion = torch.nn.CrossEntropyLoss()

params_to_update = []

update_param_names = ["classifier.6.weight", "classifier.6.bias"]

for name, param in net.named_parameters():
    if name in update_param_names:
        param.requires_grad = True
        params_to_update.append(param)
        print(name)
    else:
        param.requires_grad = False

print("--------")
print(params_to_update)

optimizer = torch.optim.SGD(params=params_to_update, lr=0.01, momentum=0.9)

print(optimizer)

torch.set_num_threads(12)



def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    print("-------------start train model-----------------")

    for epoch in range(num_epochs):
        print("Epoch {0}/{1}".format(epoch+1,num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0

            if(epoch == 0) and (phase == 'train'):
                continue

            for inputs, labels, in tqdm(dataloaders_dict[phase]):

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                    epoch_loss += loss.item() * inputs.size(0)
                    epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)

            print('{} Loss: {:.4f} Acc:{:.4f}'.format(phase, epoch_loss, epoch_acc))



train_model(net, dataloaders_dict=dataloaders_dict, criterion=criterion, optimizer=optimizer, num_epochs=k)


path = "./model.pth"

torch.save(net, path)
print("save to file:{0}", path)
