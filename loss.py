import torch
from pytorch_msssim import SSIM
import ciede2000


def total_variation_loss(x):
    ps = 256
    a = torch.square(x[:, :, :ps-1, :ps-1] - x[:, :, 1:, :ps-1])
    b = torch.square(x[:, :, :ps-1, :ps-1] - x[:, :, :ps-1, 1:])
    return torch.mean(a + b)


def mean_norm(output_b):
    x_b_mean = torch.mean(output_b, dim=(1, 2, 3))
    mean_loss = torch.mean(torch.abs(0.5 - x_b_mean))
    return mean_loss


def dE_loss(img1, img2):
    d_map = ciede2000.ciede2000_diff(ciede2000.rgb2lab_diff(img1, img1.device),
                                    ciede2000.rgb2lab_diff(img2, img2.device),
                                    img1.device).unsqueeze(1)
    color_dis = torch.norm(d_map.view(img1.shape[0], -1), dim=1)
    color_loss = color_dis.sum()
    return color_loss


class loss_2out():

    @staticmethod
    def compute(img_in_list, img_pred_list_a, img_pred_list_b):

        mae = torch.nn.L1Loss()
        mae_loss1 = mae(img_pred_list_a[0] * img_pred_list_b[0], img_in_list[0])
        mae_loss2 = mae(img_pred_list_a[1] * img_pred_list_b[1], img_in_list[1])
        mae_loss3 = mae(img_pred_list_a[2] * img_pred_list_b[2], img_in_list[2])
        mae_loss = mae_loss1 + mae_loss2 + mae_loss3 

        mae_loss1_sub1 = mae(img_pred_list_a[0] * img_pred_list_b[1], img_in_list[0])
        mae_loss1_sub2 = mae(img_pred_list_a[0] * img_pred_list_b[2], img_in_list[0])
        mae_loss2_sub1 = mae(img_pred_list_a[1] * img_pred_list_b[0], img_in_list[1])
        mae_loss2_sub2 = mae(img_pred_list_a[1] * img_pred_list_b[2], img_in_list[1])
        mae_loss3_sub1 = mae(img_pred_list_a[2] * img_pred_list_b[0], img_in_list[2])
        mae_loss3_sub2 = mae(img_pred_list_a[2] * img_pred_list_b[1], img_in_list[2])
        mae_loss_sub = mae_loss1_sub1 + mae_loss1_sub2 \
                         + mae_loss2_sub1 + mae_loss2_sub2 \
                         + mae_loss3_sub1 + mae_loss3_sub2 

        ssim_module = SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=True)
        ssim_loss_1 = 1 - ssim_module(img_pred_list_a[0] * img_pred_list_b[0], img_in_list[0])
        ssim_loss_2 = 1 - ssim_module(img_pred_list_a[1] * img_pred_list_b[1], img_in_list[1])
        ssim_loss_3 = 1 - ssim_module(img_pred_list_a[2] * img_pred_list_b[2], img_in_list[2])
        ssim_loss = torch.mean(ssim_loss_1 + ssim_loss_2 + ssim_loss_3)

        ssim_loss_1_sub1 = 1 - ssim_module(img_pred_list_a[0] * img_pred_list_b[1], img_in_list[0])
        ssim_loss_1_sub2 = 1 - ssim_module(img_pred_list_a[0] * img_pred_list_b[2], img_in_list[0])
        ssim_loss_2_sub1 = 1 - ssim_module(img_pred_list_a[1] * img_pred_list_b[0], img_in_list[1])
        ssim_loss_2_sub2 = 1 - ssim_module(img_pred_list_a[1] * img_pred_list_b[2], img_in_list[1])
        ssim_loss_3_sub1 = 1 - ssim_module(img_pred_list_a[2] * img_pred_list_b[0], img_in_list[2])
        ssim_loss_3_sub2 = 1 - ssim_module(img_pred_list_a[2] * img_pred_list_b[1], img_in_list[2])
        ssim_loss_sub = torch.mean(ssim_loss_1_sub1 + ssim_loss_1_sub2
                        + ssim_loss_2_sub1 + ssim_loss_2_sub2
                        + ssim_loss_3_sub1 + ssim_loss_3_sub2)

        mae_loss1_b = mae(img_pred_list_b[0], img_pred_list_b[1])
        mae_loss2_b = mae(img_pred_list_b[1], img_pred_list_b[2])
        mae_loss3_b = mae(img_pred_list_b[0], img_pred_list_b[2])
        mae_loss_b = mae_loss1_b + mae_loss2_b + mae_loss3_b

        bmean_loss_list = list(map(mean_norm, img_pred_list_b))
        bmean_loss = bmean_loss_list[0] + bmean_loss_list[1] + bmean_loss_list[2]

        tv_loss_list_a = list(map(total_variation_loss, img_pred_list_a))
        tv_loss_a = torch.mean(tv_loss_list_a[0] + tv_loss_list_a[1] + tv_loss_list_a[2])


        de_loss1 = dE_loss(img_pred_list_a[0] * img_pred_list_b[0], img_in_list[0])
        de_loss2 = dE_loss(img_pred_list_a[1] * img_pred_list_b[1], img_in_list[1])
        de_loss3 = dE_loss(img_pred_list_a[2] * img_pred_list_b[2], img_in_list[2])
        de_loss = (de_loss1 + de_loss2 + de_loss3) / (256*256)

        de_loss1_sub1 = mae(img_pred_list_a[0] * img_pred_list_b[1], img_in_list[0])
        de_loss1_sub2 = mae(img_pred_list_a[0] * img_pred_list_b[2], img_in_list[0])
        de_loss2_sub1 = mae(img_pred_list_a[1] * img_pred_list_b[0], img_in_list[1])
        de_loss2_sub2 = mae(img_pred_list_a[1] * img_pred_list_b[2], img_in_list[1])
        de_loss3_sub1 = mae(img_pred_list_a[2] * img_pred_list_b[0], img_in_list[2])
        de_loss3_sub2 = mae(img_pred_list_a[2] * img_pred_list_b[1], img_in_list[2])
        de_loss_sub = (de_loss1_sub1 + de_loss1_sub2 \
                         + de_loss2_sub1 + de_loss2_sub2 \
                         + de_loss3_sub1 + de_loss3_sub2)  / (256*256*2)

        return 3*(mae_loss + mae_loss_sub) + (ssim_loss + ssim_loss_sub) \
               + mae_loss_b + bmean_loss + 10*tv_loss_a + 2*(de_loss + de_loss_sub)


class loss_2out_wb():

    @staticmethod
    def compute(img_in_list, img_wb_list, img_pred_list_a, img_pred_list_b, img_pred_list_wb):

        mae = torch.nn.L1Loss()
        mae_loss1 = mae(img_pred_list_a[0] * img_pred_list_b[0], img_in_list[0])
        mae_loss2 = mae(img_pred_list_a[1] * img_pred_list_b[1], img_in_list[1])
        mae_loss3 = mae(img_pred_list_a[2] * img_pred_list_b[2], img_in_list[2])
        mae_loss = mae_loss1 + mae_loss2 + mae_loss3

        mae_loss1_sub1 = mae(img_pred_list_a[0] * img_pred_list_b[1], img_in_list[0])
        mae_loss1_sub2 = mae(img_pred_list_a[0] * img_pred_list_b[2], img_in_list[0])
        mae_loss2_sub1 = mae(img_pred_list_a[1] * img_pred_list_b[0], img_in_list[1])
        mae_loss2_sub2 = mae(img_pred_list_a[1] * img_pred_list_b[2], img_in_list[1])
        mae_loss3_sub1 = mae(img_pred_list_a[2] * img_pred_list_b[0], img_in_list[2])
        mae_loss3_sub2 = mae(img_pred_list_a[2] * img_pred_list_b[1], img_in_list[2])
        mae_loss_sub = mae_loss1_sub1 + mae_loss1_sub2 \
                         + mae_loss2_sub1 + mae_loss2_sub2 \
                         + mae_loss3_sub1 + mae_loss3_sub2

        ssim_module = SSIM(data_range=1, size_average=True, channel=3, nonnegative_ssim=True)
        ssim_loss_1 = 1 - ssim_module(img_pred_list_a[0] * img_pred_list_b[0], img_in_list[0])
        ssim_loss_2 = 1 - ssim_module(img_pred_list_a[1] * img_pred_list_b[1], img_in_list[1])
        ssim_loss_3 = 1 - ssim_module(img_pred_list_a[2] * img_pred_list_b[2], img_in_list[2])
        ssim_loss = torch.mean(ssim_loss_1 + ssim_loss_2 + ssim_loss_3)

        ssim_loss_1_sub1 = 1 - ssim_module(img_pred_list_a[0] * img_pred_list_b[1], img_in_list[0])
        ssim_loss_1_sub2 = 1 - ssim_module(img_pred_list_a[0] * img_pred_list_b[2], img_in_list[0])
        ssim_loss_2_sub1 = 1 - ssim_module(img_pred_list_a[1] * img_pred_list_b[0], img_in_list[1])
        ssim_loss_2_sub2 = 1 - ssim_module(img_pred_list_a[1] * img_pred_list_b[2], img_in_list[1])
        ssim_loss_3_sub1 = 1 - ssim_module(img_pred_list_a[2] * img_pred_list_b[0], img_in_list[2])
        ssim_loss_3_sub2 = 1 - ssim_module(img_pred_list_a[2] * img_pred_list_b[1], img_in_list[2])
        ssim_loss_sub = torch.mean(ssim_loss_1_sub1 + ssim_loss_1_sub2
                        + ssim_loss_2_sub1 + ssim_loss_2_sub2
                        + ssim_loss_3_sub1 + ssim_loss_3_sub2)

        de_loss1 = dE_loss(img_pred_list_a[0] * img_pred_list_b[0], img_in_list[0])
        de_loss2 = dE_loss(img_pred_list_a[1] * img_pred_list_b[1], img_in_list[1])
        de_loss3 = dE_loss(img_pred_list_a[2] * img_pred_list_b[2], img_in_list[2])
        de_loss = (de_loss1 + de_loss2 + de_loss3) / (256*256)

        de_loss1_sub1 = mae(img_pred_list_a[0] * img_pred_list_b[1], img_in_list[0])
        de_loss1_sub2 = mae(img_pred_list_a[0] * img_pred_list_b[2], img_in_list[0])
        de_loss2_sub1 = mae(img_pred_list_a[1] * img_pred_list_b[0], img_in_list[1])
        de_loss2_sub2 = mae(img_pred_list_a[1] * img_pred_list_b[2], img_in_list[1])
        de_loss3_sub1 = mae(img_pred_list_a[2] * img_pred_list_b[0], img_in_list[2])
        de_loss3_sub2 = mae(img_pred_list_a[2] * img_pred_list_b[1], img_in_list[2])
        de_loss_sub = (de_loss1_sub1 + de_loss1_sub2 \
                         + de_loss2_sub1 + de_loss2_sub2 \
                         + de_loss3_sub1 + de_loss3_sub2)  / (256*256*2)

        mae_loss1_b = mae(img_pred_list_b[0], img_pred_list_b[1])
        mae_loss2_b = mae(img_pred_list_b[1], img_pred_list_b[2])
        mae_loss3_b = mae(img_pred_list_b[0], img_pred_list_b[2])
        mae_loss_b = mae_loss1_b + mae_loss2_b + mae_loss3_b

        bmean_loss_list = list(map(mean_norm, img_pred_list_b))
        bmean_loss = bmean_loss_list[0] + bmean_loss_list[1] + bmean_loss_list[2]

        tv_loss_list_a = list(map(total_variation_loss, img_pred_list_a))
        tv_loss_a = torch.mean(tv_loss_list_a[0] + tv_loss_list_a[1] + tv_loss_list_a[2])

        wb_loss1 = mae(img_pred_list_wb[0], img_wb_list[0])
        wb_loss2 = mae(img_pred_list_wb[1], img_wb_list[1])
        wb_loss3 = mae(img_pred_list_wb[2], img_wb_list[2])
        wb_loss = wb_loss1 + wb_loss2 + wb_loss3

        return 3*(mae_loss + mae_loss_sub) + (ssim_loss + ssim_loss_sub) + 2*(de_loss + de_loss_sub) \
               + 3*mae_loss_b + bmean_loss \
               + 20*wb_loss + 10*tv_loss_a
