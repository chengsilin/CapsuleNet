import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from data_utils import load_mnist
from model import margin_loss, CapsuleNet
import argparse


def train(args):
    # load data
    train_dataloader, test_dataloader = load_mnist()

    # load model
    model = CapsuleNet(args).cuda()

    # define loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    # start training
    for epooch in range(args.epochs):
        for (i, data) in tqdm(enumerate(train_dataloader), total=len(train_dataloader), smoothing=0.9):
            img, label = data
            img, label = img.cuda(), label.cuda()
            label = F.one_hot(label, num_classes=args.num_class).float()
            optimizer.zero_grad()

            pred, reon = model(img, label)
            loss = margin_loss(label, pred, reon, img, args.lam_recon)
            loss.backward()
            optimizer.step()
        model.eval()
        correct = 0
        for (i, data) in tqdm(enumerate(test_dataloader), total=len(test_dataloader), smoothing=True):
            img, label = data
            img, label = img.cuda(), label.cuda()
            label = F.one_hot(label, num_classes=args.num_class).float()
            pred, reon = model(img, label)
            y_pred = pred.data.max(dim=1)[1]
            y_true = label.data.max(dim=1)[1]
            correct += (y_pred.eq(y_true).cpu().sum())
        print(correct)
        print(len(test_dataloader.dataset))
        OA = correct.data.item() / len(test_dataloader.dataset)
        print('Test acc:', OA)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Capsule Network on MNIST.")
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--input_size', default=[1, 28, 28], type=list)
    parser.add_argument('--num_class', default=10, type=int)
    parser.add_argument('--routing_num', default=3, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.9, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")
    parser.add_argument('--lam_recon', default=0.392, type=float,
                        help="The coefficient for the loss of decoder")
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")
    parser.add_argument('--shift_fraction', default=0.1, type=float,
                        help="Fraction of pixels to shift at most in each direction.")
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='./result')
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")
    parser.add_argument('--digit', default=5, type=int,
                        help="Digit to manipulate")
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    args = parser.parse_args()
    print(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    train(args)
