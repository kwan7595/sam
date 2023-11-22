import argparse
import torch
import torch.nn as nn
import os 
from model.wide_res_net import WideResNet
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar
from utility.log import Log
from utility.initialize import initialize
from utility.step_lr import StepLR
from utility.bypass_bn import enable_running_stats, disable_running_stats
from utility.meters import get_meters,Meter,ScalarMeter,flush_scalar_meters
import sys; sys.path.append("..")
from sam import SAM
from tensorboardX import SummaryWriter



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size used in the training and validation loop.")
    parser.add_argument("--depth", default=16, type=int, help="Number of layers.")
    parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")
    parser.add_argument("--epochs", default=200, type=int, help="Total number of epochs.")
    parser.add_argument("--label_smoothing", default=0.1, type=float, help="Use 0.0 for no label smoothing.")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="Base learning rate at the start of the training.")
    parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
    parser.add_argument("--threads", default=8, type=int, help="Number of CPU threads for dataloaders.")
    parser.add_argument("--rho", default=2.0, type=int, help="Rho parameter for SAM.")
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="L2 weight decay.")
    parser.add_argument("--width_factor", default=8, type=int, help="How many times wider compared to normal ResNet.")
    args = parser.parse_args()

    initialize(args, seed=42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = Cifar(args.batch_size, args.threads)
    log = Log(log_each=10)
    model = WideResNet(args.depth, args.width_factor, args.dropout, in_channels=3, labels=10).to(device)
    model = nn.DataParallel(model) 
    writer = SummaryWriter(log_dir = "./runs") # directory for tensorboard logs
    log_dir = "./log" # directory for model checkpoints
    train_meters = get_meters("train",model)
    val_meters = get_meters("val",model)
    val_meters["best_val"] = ScalarMeter("best_val")
    best_val = 0.0

    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), base_optimizer, rho=args.rho, adaptive=args.adaptive, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, args.learning_rate, args.epochs)

    for epoch in range(args.epochs):
        model.train()
        log.train(len_dataset=len(dataset.train))

        for batch in dataset.train:
            inputs, targets = (b.to(device) for b in batch)

            # first forward-backward step
            enable_running_stats(model)
            predictions = model(inputs)
            loss = smooth_crossentropy(predictions, targets, smoothing=args.label_smoothing)
            train_meters["CELoss"].cache(loss.cpu().detach().numpy())
            loss.mean().backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward step
            disable_running_stats(model)
            smooth_crossentropy(model(inputs), targets, smoothing=args.label_smoothing).mean().backward()
            optimizer.second_step(zero_grad=True)

            with torch.no_grad():
                correct = torch.argmax(predictions.data, 1) == targets
                for k in 5:
                    correct_k = correct[:k].float().sum(0)
                    acc_list = list(correct_k.cpu().detach().numpy())
                    train_meters["top{}_accuracy".format(k)].cache_list(acc_list)
                log(model, loss.cpu(), correct.cpu(), scheduler.lr())
                scheduler(epoch)

        results = flush_scalar_meters(train_meters)
        for k, v in results.items():
            if k != "best_val":
                writer.add_scalar("train" + "/" + k, v, epoch)

        model.eval()
        log.eval(len_dataset=len(dataset.test))

        val_meters["best_val"].cache(best_val)
        with torch.no_grad():
            for batch in dataset.test:
                inputs, targets = (b.to(device) for b in batch)

                predictions = model(inputs)
                loss = smooth_crossentropy(predictions, targets)
                val_meters["CELoss"].cache(loss.cpu().detach().numpy())
                correct = torch.argmax(predictions, 1) == targets
                for k in 5:
                    correct_k = correct[:k].float().sum(0)
                    acc_list = list(correct_k.cpu().detach().numpy())
                    val_meters["top{}_accuracy".format(k)].cache_list(acc_list)
                log(model, loss.cpu(), correct.cpu())
                
        results = flush_scalar_meters(val_meters)
        if results["top1_accuracy"] > best_val:
            best_val = results["top1_accuracy"]
            torch.save(model, os.path.join(log_dir, "best.pth"))
            print("New best validation top1 accuracy: {:.3f}".format(best_val))
        writer.add_scalar("val/best_val", best_val, epoch)

        if epoch ==0 or (epoch+1) % 10 == 0:
            torch.save(
                model,
                os.path.join("dir","epoch_{}.pth".format(epoch))
            )

    log.flush()
