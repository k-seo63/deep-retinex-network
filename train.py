import logging
import os
import sys
import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from model import UNet_retinex
import math
from dataset import BasicDataset
from loss import loss_2out, loss_2out_wb
from torch.utils.data import DataLoader, random_split


def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              lrdf,
              lrdp,
              fold,
              chkpointperiod,
              patchsz,
              evnum,
              validationFrequency,
              dir_img,
              save_cp=True):

    dir_checkpoint = f'checkpoints_{fold}/'
    dataset = BasicDataset(dir_img, patch_size=patchsz)
    n_val = int(len(dataset) * 0.00)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                pin_memory=True, 
                                num_workers=4)
    val_loader = DataLoader(val, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=4, 
                            pin_memory=True, 
                            drop_last=True)

    global_step = 0
    best_score = {'best': math.inf, 'count': 0}

    logging.info(f'''Starting training:
        Epochs:          {epochs} epochs
        Batch size:      {batch_size}
        Patch size:      {patchsz} x {patchsz}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation Frq.: {validationFrequency}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), 
                            lr=lr, 
                            betas=(0.9, 0.999), 
                            eps=1e-08, 
                            weight_decay=0.00001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                            lrdp, 
                                            gamma=lrdf, 
                                            last_epoch=-1)

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs_ = batch['input']
                wb_ = batch['input_wb']
                img_in_list = []
                img_wb_list = []
                img_pred_list_a = []
                img_pred_list_b = []
                img_pred_list_wb = []

                for j in range(evnum):
                    imgs = imgs_[:, (j * 3): 3 + (j * 3), :, :]
                    imgs = imgs.to(device=device, dtype=torch.float32)

                    img_pred_a, img_pred_b, img_pred_wb = net(imgs)

                    img_in_list.append(imgs)
                    img_wb_list.append(wb_[:, (j * 3): 3 + (j * 3)].to(device=device, dtype=torch.float32))
                    img_pred_list_a.append(img_pred_a)
                    img_pred_list_b.append(img_pred_b)
                    img_pred_list_wb.append(torch.squeeze(img_pred_wb))

                loss = loss_2out_wb.compute(img_in_list, 
                                            img_wb_list, 
                                            img_pred_list_a, 
                                            img_pred_list_b, 
                                            img_pred_list_wb)

                epoch_loss += loss.item()

                pbar.set_postfix(**{'loss(batch)': loss.item()})
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pbar.update(np.ceil(imgs.shape[0]))
                global_step += 1

        scheduler.step()
        if save_cp and (epoch + 1) % chkpointperiod == 0:
            if not os.path.exists(dir_checkpoint):
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')

            torch.save(net.state_dict(), dir_checkpoint + f'epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved!')

        if (epoch + 1) % validationFrequency == 0:
            val_score = vald_net(net, val_loader, device)
            logging.info('Validation Loss: {}'.format(val_score))

            best_score = val_score_comparison(val_score, best_score)
            if best_score['count'] == 0:
                if not os.path.exists(dir_checkpoint):
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                torch.save(net.state_dict(), dir_checkpoint + 'best_epoch.pth')
                logging.info(f'Best epoch {epoch + 1} saved!')

            if best_score['count'] > 5:
                logging.info('Stop training')

    if not os.path.exists('models'):
        os.mkdir('models')
        logging.info('Created trained models directory')
    torch.save(net.state_dict(), 'models/' + 'trained_model.pth')
    logging.info('Saved trained model!')
    logging.info('End of training')


def vald_net(net, loader, device, evnum=3):
    net.eval()
    n_val = len(loader) + 1
    loss_val = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs_ = batch['input']
            img_pred_list_a = []
            img_pred_list_b = []
            img_in_list = []

            for j in range(evnum):
                imgs = imgs_[:, (j * 3): 3 + (j * 3), :, :]
                imgs = imgs.to(device=device, dtype=torch.float32)

                with torch.no_grad():
                    img_pred_a, img_pred_b = net(imgs)
                    img_in_list.append(imgs)
                    img_pred_list_a.append(img_pred_a)
                    img_pred_list_b.append(img_pred_b)

            loss = loss_2out.compute(img_in_list, 
                                    img_pred_list_a, 
                                    img_pred_list_b)
            loss_val = loss_val + loss

            pbar.update(np.ceil(imgs.shape[0]))

    net.train()
    return loss_val / n_val


def val_score_comparison(current, best):
    if current < best['best']:
        best['best'] = current
        best['count'] = 0
    else:
        best['count'] += 1
    return best


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info('Training of Deep Retinex')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet_retinex()
    net.to(device=device)
    # net = torch.nn.DataParallel(net)

    try:
        train_net(net=net,
                  epochs=50,
                  batch_size=8,
                  lr=0.001,
                  lrdf=0.1,
                  lrdp=20,
                  device=device,
                  fold=0,
                  chkpointperiod=10,
                  val_percent=10/100,
                  validationFrequency=5,
                  patchsz=256,
                  evnum=3,
                  dir_img='./imgs/**'
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'bkupCheckPoint.pth')
        logging.info('Saved interrupt checkpoint backup')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
