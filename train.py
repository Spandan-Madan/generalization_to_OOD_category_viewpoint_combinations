from __future__ import print_function, division

print("I am starting")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
from PIL import ImageFile
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import pickle
import sys

sys.path.append("../../res/")
from models.models import get_model
from loader.loader import get_loader

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--num_epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--arch", type=str)
parser.add_argument("--save_file_suffix", type=str)
parser.add_argument("--start_checkpoint_path", type=str)
parser.add_argument("--task", type=srt, default="combined")
args = parser.parse_args()


print(args, flush=True)
sys.stdout.flush()
DATASET_NAME = args.dataset_name
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
ARCH = args.arch
SAVE_FILE_SUFFIX = args.save_file_suffix
TASK = args.task

if args.start_checkpoint_path:
    checkpoint_model_name = args.start_checkpoint_path.split("/")[-1].split(".pt")[0]
    SAVE_FILE_SUFFIX = "%s_%s" % (SAVE_FILE_SUFFIX, checkpoint_model_name)
# In[36]:
train_file_name = __file__.split(".")[0]

if TASK == "combined":
    SAVE_FILE = "logs/%s_%s_%s_%s_%s_%s.out" % (
        train_file_name,
        ARCH,
        DATASET_NAME,
        NUM_EPOCHS,
        BATCH_SIZE,
        SAVE_FILE_SUFFIX,
    )
else:
    SAVE_FILE = "logs/%s_%s_%s_%s_%s_%s_%s.out" % (
        train_file_name,
        TASK,
        ARCH,
        DATASET_NAME,
        NUM_EPOCHS,
        BATCH_SIZE,
        SAVE_FILE_SUFFIX,
    )

SAVE_HANDLER = open(SAVE_FILE, "w")

print(args, file=SAVE_HANDLER, flush=True)
sys.stdout.flush()
image_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


GPU = 1

if "ilab" in DATASET_NAME:
    NUM_CLASSES = (6, 6, 6, 6)
else:
    NUM_CLASSES = (5, 5, 5, 5)


model = get_model(ARCH, NUM_CLASSES)

if args.start_checkpoint_path:
    print("Loading from %s" % args.start_checkpoint_path)
    model = torch.load(args.start_checkpoint_path)

if "ilab" in DATASET_NAME:
    loader_new = get_loader("multi_attribute_loader_file_list_ilab")
else:
    loader_new = get_loader("multi_attribute_loader_file_list")

file_list_root = "/om5/user/smadan/dataset_lists_openmind"
att_path = "/om5/user/smadan/dataset_lists/combined_attributes.p"

# loader_new = get_loader('multi_attribute_loader_file_list_ilab')
# file_list_root = '/data/graphics/toyota-pytorch/biased_dataset_generalization/dataset_lists'
# #att_path = '/data/graphics/toyota-pytorch/biased_dataset_generalization/dataset_lists/merged_colors_binned_attribute.p'
# att_path = '/data/graphics/toyota-pytorch/biased_dataset_generalization/dataset_lists/combined_attributes.p'
shuffles = {"train": True, "val": True, "test": False}


################ GET FROM USER CONFIG - TODO #####################
file_lists = {}
dsets = {}
dset_loaders = {}
dset_sizes = {}
for phase in ["train", "val", "test"]:
    file_lists[phase] = "%s/%s_list_%s.txt" % (file_list_root, phase, DATASET_NAME)
    dsets[phase] = loader_new(file_lists[phase], att_path, image_transform)
    dset_loaders[phase] = torch.utils.data.DataLoader(
        dsets[phase],
        batch_size=BATCH_SIZE,
        shuffle=shuffles[phase],
        num_workers=2,
        drop_last=True,
    )
    dset_sizes[phase] = len(dsets[phase])


# In[37]:

print("Dataset sizes:", file=SAVE_HANDLER)
print(len(dsets["train"]), file=SAVE_HANDLER)
print(len(dsets["val"]), file=SAVE_HANDLER)
print(len(dsets["test"]), file=SAVE_HANDLER)
sys.stdout.flush()
# loss = nn.CrossEntropyLoss()
multi_losses = [
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
    nn.CrossEntropyLoss(),
]

# In[9]:

if GPU:
    #     loss.cuda()
    model.cuda()
    if ARCH == "MULTITASKRESNET":
        model.fcs = [i.cuda() for i in model.fcs]

optimizer = optim.Adam(model.parameters(), lr=0.001)

# In[10]:


# In[39]:


def weight_scheduler(epoch_num):
    if TASK == "rotation":
        return [0.0, 1.0, 0.0, 0.0]
    if TASK == "car_model":
        return [0.0, 0.0, 0.0, 1.0]
    elif TASK == "combined":
        return [0.0, 1.0, 0.0, 1.0]


best_model = model
best_model_gm = model
best_acc = 0.0

losses = {}
accuracies = {}

losses["train"] = []
losses["val"] = []

accuracies["train"] = []
accuracies["val"] = []

best_val_loss = 100

for epoch in range(NUM_EPOCHS):
    print("Epoch %s" % epoch, file=SAVE_HANDLER, flush=True)
    sys.stdout.flush()
    weights = weight_scheduler(epoch)
    for phase in ("train", "val"):
        print("%s phase" % phase, file=SAVE_HANDLER, flush=True)
        sys.stdout.flush()
        iters = 0
        phase_epoch_corrects = [0, 0, 0, 0]
        phase_epoch_loss = 0
        if phase == "train":
            model.train()
            torch.set_grad_enabled(True)
        else:
            print("model eval", file=SAVE_HANDLER)
            sys.stdout.flush()
            model.eval()
            torch.set_grad_enabled(False)

        for data in dset_loaders[phase]:
            inputs, labels_all, paths = data
            if GPU:
                inputs = Variable(inputs.float().cuda())

            optimizer.zero_grad()
            model_outs = model(inputs)

            calculated_loss = 0

            batch_corrects = [0, 0, 0, 0]
            for i in range(4):
                labels = labels_all[:, i]
                if GPU:
                    labels = Variable(labels.long().cuda())

                loss = multi_losses[i]
                outputs = model_outs[i]
                calculated_loss += weights[i] * loss(outputs, labels)
                _, preds = torch.max(outputs.data, 1)
                batch_corrects[i] = torch.sum(preds == labels.data)
                phase_epoch_corrects[i] += batch_corrects[i]

            phase_epoch_loss += calculated_loss

            if phase == "train":
                calculated_loss.backward()
                optimizer.step()

            if iters % 50 == 0:
                print("Epoch %s, Iters %s" % (epoch, iters), file=SAVE_HANDLER)
                sys.stdout.flush()
            iters += 1

        epoch_loss = phase_epoch_loss / dset_sizes[phase]
        epoch_accs = [float(i) / dset_sizes[phase] for i in phase_epoch_corrects]
        gm_epoch_accs = 1
        for i in range(4):
            gm_epoch_accs = gm_epoch_accs * epoch_accs[i]
        if gm_epoch_accs > best_acc:
            best_acc = gm_epoch_accs
            best_model_gm = model

        print("Epoch loss: %s" % epoch_loss.item(), file=SAVE_HANDLER)
        print("Epoch accs: ", epoch_accs, file=SAVE_HANDLER)
        sys.stdout.flush()

        losses[phase].append(epoch_loss.item())
        accuracies[phase].append(epoch_accs)

        if phase == "val":
            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_model = model

            gm_epoch_accs = 1
            for i in range(2):
                gm_epoch_accs = gm_epoch_accs * epoch_accs[i]

            if gm_epoch_accs > best_acc:
                best_acc = gm_epoch_accs
                best_model_gm = model

if TASK == "combined":
    with open(
        "saved_models/%s_model_%s_%s_%s.pt"
        % (train_file_name, ARCH, DATASET_NAME, SAVE_FILE_SUFFIX),
        "wb",
    ) as F:
        torch.save(best_model, F)

    with open(
        "losses/%s_losses_%s_%s_%s.pt"
        % (train_file_name, ARCH, DATASET_NAME, SAVE_FILE_SUFFIX),
        "wb",
    ) as F:
        pickle.dump(losses, F)

    with open(
        "accuracies/%s_accuracies_%s_%s_%s.pt"
        % (train_file_name, ARCH, DATASET_NAME, SAVE_FILE_SUFFIX),
        "wb",
    ) as F:
        pickle.dump(accuracies, F)
else:
    with open(
        "saved_models/%s_%s_model_%s_%s_%s.pt"
        % (train_file_name, TASK, ARCH, DATASET_NAME, SAVE_FILE_SUFFIX),
        "wb",
    ) as F:
        torch.save(best_model, F)

    with open(
        "losses/%s_%s_losses_%s_%s_%s.pt"
        % (train_file_name, TASK, ARCH, DATASET_NAME, SAVE_FILE_SUFFIX),
        "wb",
    ) as F:
        pickle.dump(losses, F)

    with open(
        "accuracies/%s_%s_accuracies_%s_%s_%s.pt"
        % (train_file_name, TASK, ARCH, DATASET_NAME, SAVE_FILE_SUFFIX),
        "wb",
    ) as F:
        pickle.dump(accuracies, F)

for phase in ["test"]:
    print("%s phase" % phase, file=SAVE_HANDLER)
    phase_epoch_corrects = 0
    model.eval()
    torch.set_grad_enabled(False)
    test_epoch_corrects = [0, 0, 0, 0]
    for data in dset_loaders[phase]:
        inputs, labels_all, paths = data
        if GPU:
            inputs = Variable(inputs.float().cuda())
        model_outs = best_model(inputs)

        test_batch_corrects = [0, 0, 0, 0]
        for i in range(4):
            labels = labels_all[:, i]
            if GPU:
                labels = Variable(labels.long().cuda())
            outputs = model_outs[i]

            _, preds = torch.max(outputs.data, 1)
            test_batch_corrects[i] = torch.sum(preds == labels.data)
            test_epoch_corrects[i] += test_batch_corrects[i]

    test_epoch_accs = [float(i) / dset_sizes[phase] for i in test_epoch_corrects]
    print("Epoch acc: ", test_epoch_accs, file=SAVE_HANDLER)

if TASK == "combined":
    with open(
        "accuracies/%s_test_accuracies_%s_%s_%s.pt"
        % (train_file_name, ARCH, DATASET_NAME, SAVE_FILE_SUFFIX),
        "wb",
    ) as F:
        pickle.dump(test_epoch_accs, F)
else:
    with open(
        "accuracies/%s_%s_test_accuracies_%s_%s_%s.pt"
        % (train_file_name, TASK, ARCH, DATASET_NAME, SAVE_FILE_SUFFIX),
        "wb",
    ) as F:
        pickle.dump(test_epoch_accs, F)
