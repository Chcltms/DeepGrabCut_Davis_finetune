import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict
import scipy.misc as sm
import numpy as np

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import upsample

# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
#from dataloaders.combine_dbs import CombineDBs as combine_dbs
#from dataloaders import pascal, sbd
#from networks import deeplab_resnet as resnet
from dataloaders.Davis_refine_loader import Davis_refine_loader
from layers.loss import class_balanced_cross_entropy_loss
from dataloaders import custom_transforms as tr
from dataloaders.utils import generate_param_report
from deeplab_resnet import DeepLabv3_plus
     

gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
use_sbd = True
nEpochs = 200  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
#classifier = 'psp'  # Head classifier to use
p['trainBatch'] = 24  # Training batch size
testBatch = 4  # Testing batch size
useTest = True  # See evolution of the test set when training
nTestInterval = 10  # Run on test set every nTestInterval epochs
snapshot = 1  # Store a model every snapshot epochs
nInputChannels = 4  # Number of input channels (RGB + Distance Map of bounding box)
zero_pad_crop = True  # Insert zero padding when cropping the image
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 1e-6  # Learning rate
p['wd'] = 0.0005  # Weight decay
p['momentum'] = 0.9  # Momentum
p['epoch_size'] = 20

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    run_id = 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))

modelName = 'deepgc_pascal'

net = DeepLabv3_plus(n_classes=1, nInputChannels=4)
model_name = '/home/ningxinLin/deeplabv3plus_coco/deeplab_coco_4_channel.pth'
checkpoint = torch.load(model_name)
net.load_state_dict(checkpoint)
net = torch.nn.DataParallel(net).cuda()


if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    #optimizer = optim.Adam(net.parameters(), lr=p['lr'])
    #p['optimizer'] = str(optimizer)

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.ScaleNRotate(rots=(-15, 15), scales=(.75, 1.25)),
        tr.FixedResize(resolutions={'image': (450, 450), 'gt': (450, 450)}),
        tr.DistanceMap(v=0.3, elem='gt'),
        tr.ConcatInputs(elems=('image', 'distance_map')),
        tr.ToTensor()])


    davis_train = Davis_refine_loader(transform=composed_transforms_tr)


    trainloader = DataLoader(davis_train, batch_size=p['trainBatch'], shuffle=True, num_workers=16)

    generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    print("Training Network")

    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()
        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, gts = sample_batched['concat'], sample_batched['gt']

            # Forward-Backward of the mini-batch
            inputs, gts = Variable(inputs, requires_grad=True), Variable(gts)
            if gpu_id >= 0:
                inputs, gts = inputs.cuda(), gts.cuda()

            output = net.forward(inputs)
            output = upsample(output, size=(450, 450), mode='bilinear', align_corners=True)


            # Compute the losses, side outputs and fuse
            loss = class_balanced_cross_entropy_loss(output, gts, size_average=False, batch_average=True)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == num_img_tr - 1:
                running_loss_tr = running_loss_tr / num_img_tr
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

        # Save the model
        if (epoch % snapshot) == snapshot - 1 and epoch != 0:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
        
        if (epoch % p['epoch_size']) == p['epoch_size'] - 1:
            p['lr'] = p['lr'] * 0.5
            print('(poly lr policy) learning rate: ', p['lr'])
            optimizer = optim.Adam(net.parameters(), lr=p['lr'])
        # One testing epoch

    writer.close()
