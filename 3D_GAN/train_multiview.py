import torch
from torch import optim
from torch import nn
from collections import OrderedDict
from utils import make_hyparam_string, plot_generated_vs_real, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ
import utils
import os
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from utils import ShapeNetMultiviewDataset, var_or_cuda
from model import _G, _D, _E_MultiView
from lr_sh import MultiStepLR
import time  # Yeni import

plt.switch_backend("TkAgg")

def KLLoss(z_mu,z_var):
    return (- 0.5 * torch.sum(1 + z_var - torch.pow(z_mu, 2) - torch.exp(z_var)))

def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) between two 3D volumes
    pred: predicted volume (batch_size, 1, D, H, W)
    target: target volume (batch_size, 1, D, H, W)
    threshold: value for binarizing the predictions
    """
    pred = (pred > threshold).float()
    target = (target > threshold).float()
    
    intersection = torch.sum(pred * target, dim=(1,2,3,4))
    union = torch.sum(pred, dim=(1,2,3,4)) + torch.sum(target, dim=(1,2,3,4)) - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)  # Adding small epsilon to avoid division by zero
    return torch.mean(iou)

def calculate_metrics(predictions, labels, threshold=0.5):
    """
    Calculate precision, recall and F1 score for discriminator predictions
    predictions: discriminator output
    labels: true labels (0 for fake, 1 for real)
    """
    predictions = (predictions >= threshold).float()
    labels = labels >= 0.5  # Convert soft labels to binary

    true_positives = torch.sum(predictions * labels)
    false_positives = torch.sum(predictions * (~labels))
    false_negatives = torch.sum((~predictions.bool()) * labels)
    
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return precision.item(), recall.item(), f1.item()

def train_multiview(args):
    hyparam_list = [("model", args.model_name),
                    ("cube", args.cube_len),
                    ("bs", args.batch_size),
                    ("g_lr", args.g_lr),
                    ("d_lr", args.d_lr),
                    ("z", args.z_dis),
                    ("bias", args.bias),
                    ("sl", args.soft_label), ]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    print(log_param)

    # for using tensorboard
    if args.use_tensorboard:
        import tensorflow as tf
        log_dir = args.output_dir + args.log_dir + log_param
        summary_writer = tf.summary.create_file_writer(log_dir)

        def inject_summary(summary_writer, tag, value, step):
            with summary_writer.as_default():
                tf.summary.scalar(tag, value, step=step)
                summary_writer.flush()

        inject_summary = inject_summary

    # datset define
    dsets_path = args.input_dir + args.data_dir + "train/"
    print(dsets_path)
    dsets = ShapeNetMultiviewDataset(dsets_path, args)
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # model define
    D = _D(args)
    G = _G(args)
    E = _E_MultiView(args)

    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    E_solver = optim.Adam(E.parameters(), lr=args.e_lr, betas=args.beta)

    if args.lrsh:
        D_scheduler = MultiStepLR(D_solver, milestones=[500, 1000])

    if torch.cuda.is_available():
        print("using cuda")
        D.cuda()
        G.cuda()
        E.cuda()

    criterion = nn.BCELoss()

    pickle_path = "." + args.pickle_dir + log_param
    read_pickle(pickle_path, G, G_solver, D, D_solver)

    for epoch in range(args.n_epochs):
        epoch_start_time = time.time()  # Epok başlangıç zamanı
        
        # Epoch başında değerleri sıfırla
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_d_total_acu = 0
        epoch_iou = 0  # IoU için yeni değişken
        epoch_precision = 0
        epoch_recall = 0
        epoch_f1 = 0
        batch_count = 0

        for i, (images, model_3d) in enumerate(dset_loaders):

            model_3d = var_or_cuda(model_3d)
            if (model_3d.size()[0] != int(args.batch_size)):
                 # print("batch_size != {} drop last incompatible batch".format(int(args.batch_size)))
                continue

            Z = generateZ(args)
            Z_vae, z_mus, z_vars = E(images)
            #Z_vae = E.reparameterize(z_mu, z_var)
            G_vae = G(Z_vae)

            real_labels = var_or_cuda(torch.ones(args.batch_size))
            fake_labels = var_or_cuda(torch.zeros(args.batch_size))

            if args.soft_label:
                real_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0.7, 1.0))
                fake_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0, 0.3))

            # ============= Train the discriminator =============#
            d_real = D(model_3d)
            d_real_loss = criterion(d_real, real_labels)

            fake = G(Z)
            d_fake = D(fake)
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss

            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            if d_total_acu <= args.d_thresh:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()

            # Discriminator metrikleri hesapla
            d_precision, d_recall, d_f1 = calculate_metrics(
                torch.cat([d_real, d_fake]), 
                torch.cat([real_labels, fake_labels])
            )

            # ============= Train the Encoder =============#
            model_3d = model_3d.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)
            recon_loss = torch.sum(torch.pow((G_vae - model_3d), 2))
            kl_loss = 0
            for i in range(args.num_views):
                kl_loss += KLLoss(z_vars[i],z_mus[i])

            kl_loss = kl_loss / args.num_views
            E_loss = recon_loss + kl_loss

            E.zero_grad()
            E_loss.backward(retain_graph=True)
            E_solver.step()
            # =============== Train the generator ===============#

            Z = generateZ(args)

            fake = G(Z)
            d_fake = D(fake)
            g_loss = criterion(d_fake, real_labels)
            g_loss += recon_loss

            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_solver.step()

            # Her batch sonunda IoU hesapla
            batch_iou = calculate_iou(G_vae, model_3d.view(-1, 1, args.cube_len, args.cube_len, args.cube_len))
            
            # Her batch sonunda değerleri topla
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_total_acu += d_total_acu.item()
            epoch_iou += batch_iou.item()  # IoU toplamı
            epoch_precision += d_precision
            epoch_recall += d_recall
            epoch_f1 += d_f1
            batch_count += 1

        # Epoch süresini hesapla
        epoch_duration = time.time() - epoch_start_time

        # Epoch sonunda ortalamaları hesapla
        avg_recon_loss = epoch_recon_loss / batch_count
        avg_kl_loss = epoch_kl_loss / batch_count
        avg_d_loss = epoch_d_loss / batch_count
        avg_g_loss = epoch_g_loss / batch_count
        avg_d_total_acu = epoch_d_total_acu / batch_count
        avg_iou = epoch_iou / batch_count  # IoU ortalaması
        avg_precision = epoch_precision / batch_count
        avg_recall = epoch_recall / batch_count
        avg_f1 = epoch_f1 / batch_count

        # =============== logging each iteration ===============#
        iteration = int(G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][0]]['step'].item())
        if args.use_tensorboard:
            log_save_path = args.output_dir + args.log_dir + log_param
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)

            info = {
                'loss/loss_D_R': d_real_loss.item(),
                'loss/loss_D_F': d_fake_loss.item(),
                'loss/loss_D': avg_d_loss,
                'loss/loss_G': avg_g_loss,
                'loss/acc_D': avg_d_total_acu,
                'loss/loss_recon': avg_recon_loss,
                'loss/loss_kl': avg_kl_loss,
                'metrics/iou': avg_iou,  # IoU metriği eklendi
                'metrics/precision': avg_precision,
                'metrics/recall': avg_recall,
                'metrics/f1': avg_f1,
                'time/epoch_duration': epoch_duration,  # Süre metriği eklendi
            }

            for tag, value in info.items():
                inject_summary(summary_writer, tag, value, epoch)

            summary_writer.flush()

        # =============== each epoch save model or save image ===============#
        print(
            'Epoch-{}, Iter-{}; Time: {:.2f}s, Avg_Recon_loss : {:.4f}, Avg_KLLoss: {:.4f}, Avg_D_loss : {:.4f}, Avg_G_loss : {:.4f}, Avg_D_acu : {:.4f}, Avg_IoU : {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, D_lr : {:.4f}'.format(
                epoch,
                iteration,
                epoch_duration, 
                avg_recon_loss,
                avg_kl_loss,
                avg_d_loss,
                avg_g_loss,
                avg_d_total_acu,
                avg_iou,
                avg_precision,
                avg_recall,
                avg_f1,
                D_solver.state_dict()['param_groups'][0]["lr"]))

        if (epoch + 1) % args.image_save_step == 0:

            samples = fake.cpu().data[:8].squeeze().numpy()

            image_path = args.output_dir + args.image_dir + log_param
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            SavePloat_Voxels(samples, image_path, iteration)
            
            # Karşılaştırmalı görselleştirme
            comparison_path = args.output_dir + args.image_dir + log_param + '/comparisons'
            if not os.path.exists(comparison_path):
                os.makedirs(comparison_path)
            plot_generated_vs_real(fake.cpu().data, model_3d.cpu().data, 
                                 comparison_path, iteration)

        if (epoch + 1) % args.pickle_step == 0:
            pickle_save_path = args.output_dir + args.pickle_dir + log_param
            save_new_pickle(pickle_save_path, iteration, G, G_solver, D, D_solver, E, E_solver)

        if args.lrsh:

            try:

                D_scheduler.step()


            except Exception as e:

                print("fail lr scheduling", e)
