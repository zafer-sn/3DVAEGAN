import torch
from torch import optim
from torch import  nn
from collections import OrderedDict
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ, calculate_iou, calculate_metrics
import utils
import os
import time  # Süre ölçümü için time modülünü ekledim
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from utils import ShapeNetPlusImageDataset, var_or_cuda
from model import _G, _D, _E
from lr_sh import  MultiStepLR
plt.switch_backend("TkAgg")

def train_vae(args):

    hyparam_list = [("model", args.model_name),
                    ("cube", args.cube_len),
                    ("bs", args.batch_size),
                    ("g_lr", args.g_lr),
                    ("d_lr", args.d_lr),
                    ("z", args.z_dis),
                    ("bias", args.bias),
                    ("sl", args.soft_label),]

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyparam_list))
    log_param = make_hyparam_string(hyparam_dict)
    print(log_param)

    # for using tensorboard
    if args.use_tensorboard:
        import tensorflow as tf

        summary_writer = tf.summary.create_file_writer(args.output_dir + args.log_dir + log_param)

        def inject_summary(summary_writer, tag, value, step):
            with summary_writer.as_default():
                tf.summary.scalar(tag, value, step=step)

        inject_summary = inject_summary


    # datset define
    dsets_path = args.input_dir + args.data_dir + "train/"
    print(dsets_path)
    dsets = ShapeNetPlusImageDataset(dsets_path, args)
    dset_loaders = torch.utils.data.DataLoader(dsets, batch_size=args.batch_size, shuffle=True, num_workers=1)

    # model define
    D = _D(args)
    G = _G(args)
    E = _E(args)

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
        # Epok başlangıç zamanını kaydet
        epoch_start_time = time.time()
        
        # Epoch ortalamaları için değerleri sıfırlama
        epoch_d_real_loss = 0
        epoch_d_fake_loss = 0
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_d_total_acu = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_iou = 0
        # Yeni metrikler için değişkenler
        epoch_precision = 0
        epoch_recall = 0
        epoch_f1 = 0
        batch_count = 0
        
        for i, (image, model_3d) in enumerate(dset_loaders):

            model_3d = var_or_cuda(model_3d)
            image = var_or_cuda(image)




            if model_3d.size()[0] != int(args.batch_size):
                #print("batch_size != {} drop last incompatible batch".format(int(args.batch_size)))
                continue

            Z = generateZ(args)
            z_mu,z_var = E(image)
            Z_vae = E.reparameterize(z_mu,z_var)
            G_vae = G(Z_vae)

            real_labels = var_or_cuda(torch.ones(args.batch_size))
            fake_labels = var_or_cuda(torch.zeros(args.batch_size))

            if args.soft_label:
                real_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0.7, 1.0))  # Changed upper bound to 1.0
                fake_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0, 0.3))

            # ============= Train the discriminator =============#
            d_real = D(model_3d)
            d_real_loss = criterion(d_real.squeeze(), real_labels)

            fake = G(Z)
            d_fake = D(fake)
            d_fake_loss = criterion(d_fake.squeeze(), fake_labels)

            d_loss = d_real_loss + d_fake_loss

            d_real_acu = torch.ge(d_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(d_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu),0))

            if d_total_acu <= args.d_thresh:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()
            
            # ============= Train the Encoder =============#
            model_3d = model_3d.view(-1, 1, args.cube_len, args.cube_len, args.cube_len)
            recon_loss = torch.sum(torch.pow((G_vae - model_3d),2))
            KLLoss = (- 0.5 * torch.sum( 1 + z_var - torch.pow(z_mu,2) - torch.exp(z_var)))  #/dim1/dim2/dim3)
            E_loss = recon_loss + KLLoss

            # IoU değerini batch için hesaplama
            batch_iou = calculate_iou(G_vae, model_3d)
            
            E.zero_grad()
            E_loss.backward()
            E_solver.step()
            # =============== Train the generator ===============#
            # Generator için yeni bir forward pass yapılıyor
            Z_new = generateZ(args)
            fake = G(Z_new)
            d_fake = D(fake)
            gan_loss = criterion(d_fake.squeeze(), real_labels)
            
            # Generator diğer kayıpları yeniden hesaplama ve detach kullanımı 
            # (bu sayede encoder gradyanları ile çakışmayı önlüyoruz)
            Z_vae_detached = Z_vae.detach()  # Gradyan akışını kes
            G_vae_new = G(Z_vae_detached)
            recon_loss_new = torch.sum(torch.pow((G_vae_new - model_3d),2))
            
            # Generator toplam kaybı
            g_loss = gan_loss + recon_loss_new

            # Generator eğitimi
            D.zero_grad()
            G.zero_grad()
            g_loss.backward()
            G_solver.step()

            # Discriminator metrikleri hesaplama
            # Real ve fake sonuçları birleştirerek precision, recall, f1 hesaplama
            all_preds = torch.cat([d_real.squeeze(), d_fake.squeeze()])
            all_labels = torch.cat([real_labels, fake_labels])
            precision, recall, f1 = calculate_metrics(all_preds, all_labels)

            # Batch sonuçlarını toplama
            epoch_d_real_loss += d_real_loss.item()
            epoch_d_fake_loss += d_fake_loss.item()
            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            epoch_d_total_acu += d_total_acu.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += KLLoss.item()
            epoch_iou += batch_iou
            # Yeni metrikleri toplama
            epoch_precision += precision
            epoch_recall += recall
            epoch_f1 += f1
            batch_count += 1

        # Epoch ortalaması hesaplama
        if batch_count > 0:
            epoch_d_real_loss /= batch_count
            epoch_d_fake_loss /= batch_count
            epoch_d_loss /= batch_count
            epoch_g_loss /= batch_count
            epoch_d_total_acu /= batch_count
            epoch_recon_loss /= batch_count
            epoch_kl_loss /= batch_count
            epoch_iou /= batch_count
            # Yeni metriklerin ortalaması
            epoch_precision /= batch_count
            epoch_recall /= batch_count
            epoch_f1 /= batch_count

        # Epok süresini hesapla (saniye cinsinden)
        epoch_duration = time.time() - epoch_start_time

        # =============== logging each iteration ===============#
        iteration = int(G_solver.state_dict()['state'][G_solver.state_dict()['param_groups'][0]['params'][0]]['step'].item())
        if args.use_tensorboard:
            log_save_path = args.output_dir + args.log_dir + log_param
            if not os.path.exists(log_save_path):
                os.makedirs(log_save_path)

            info = {
                'loss/loss_D_R': epoch_d_real_loss,
                'loss/loss_D_F': epoch_d_fake_loss,
                'loss/loss_D': epoch_d_loss,
                'loss/loss_G': epoch_g_loss,
                'loss/acc_D' : epoch_d_total_acu,
                'loss/loss_recon' : epoch_recon_loss,
                'loss/loss_kl' : epoch_kl_loss,
                'metrics/iou' : epoch_iou,
                # Yeni metrikleri loglama
                'metrics/precision': epoch_precision,
                'metrics/recall': epoch_recall,
                'metrics/f1_score': epoch_f1,
                'performance/epoch_duration': epoch_duration  # Epok süresini TensorBoard'a kaydet
            }

            for tag, value in info.items():
                inject_summary(summary_writer, tag, value, epoch)

            summary_writer.flush()

        # =============== each epoch save model or save image ===============#
        print('Epoch-{}, Iter-{}; Duration: {:.2f}s, IoU: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, Recon_loss: {:.4}, KLLoss: {:.4}, D_loss: {:.4}, G_loss: {:.4}, D_acu: {:.4}, D_lr: {:.4}'.format(
            epoch,
            iteration,
            epoch_duration,  # Epok süresini ekrana yazdır
            epoch_iou,
            epoch_precision,
            epoch_recall,
            epoch_f1,
            epoch_recon_loss,
            epoch_kl_loss,
            epoch_d_loss, 
            epoch_g_loss, 
            epoch_d_total_acu, 
            D_solver.state_dict()['param_groups'][0]["lr"]
        ))

        if (epoch + 1) % args.image_save_step == 0:

            samples = fake.cpu().data[:8].squeeze().numpy()

            image_path = args.output_dir + args.image_dir + log_param
            if not os.path.exists(image_path):
                os.makedirs(image_path)

            SavePloat_Voxels(samples, image_path, epoch)

        if (epoch + 1) % args.pickle_step == 0:
            pickle_save_path = args.output_dir + args.pickle_dir + log_param
            save_new_pickle(pickle_save_path, epoch, G, G_solver, D, D_solver,E,E_solver)

        if args.lrsh:

            try:

                D_scheduler.step()


            except Exception as e:

                print("fail lr scheduling", e)
