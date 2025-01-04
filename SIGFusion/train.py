

from tqdm import tqdm, trange
import torch.nn.functional as F
from utils import gradient, clamp
from SIGFusion_model import SIGNet
from SIGFusion_dataset import SIGFusion_dataset
import torch
from torch.utils.data import DataLoader
import pytorch_msssim

def main():

    epoch = 100
    lr_start = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fusionmodel = SIGNet()
    fusionmodel.to(device)
    optimizer = torch.optim.Adam(fusionmodel.parameters(), lr=lr_start)
    ssim_loss = pytorch_msssim.msssim

    train_dataset = SIGFusion_dataset("train")
    print("the training dataset is length:{}".format(train_dataset.length))
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = SIGFusion_dataset("eval")
    print("the training dataset is length:{}".format(test_dataset.length))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    for i in range(epoch):
        print("---------train--------", i+1)
        fusionmodel.train()
        train_tqdm = tqdm(train_loader, total=len(train_loader))
        for image_vis_mask,vis_y_image, vis_cb_image, vis_cr_image,image_ir,image_ir_mask  in train_tqdm:

            vis_y_image = vis_y_image.to(device)
            image_vis_mask = image_vis_mask.to(device)
            image_ir = image_ir.to(device)
            image_ir_mask = image_ir_mask.to(device)
            outputs = fusionmodel(image_vis_y = vis_y_image,image_ir = image_ir)

            outputs_vis_mask_cut = torch.mul(outputs, image_vis_mask)   #得到融合图像，可见光掩膜中的图像
            outputs_ir_mask_cut = torch.mul(outputs, image_ir_mask)     #得到融合图像，红外掩膜中的图像
            image_vis_mask_cut = torch.mul(vis_y_image, image_vis_mask) #得到可见光 Y通道，可见光掩膜中的图像
            image_ir_mask_cut = torch.mul(image_ir, image_ir_mask)      #得到红外图像，红外掩膜中的图像

            loss1_back_vis = F.l1_loss(outputs, vis_y_image)
            loss1_back_ir = F.l1_loss(outputs, image_ir)
            loss1_mask_cut_ir = F.l1_loss(outputs_ir_mask_cut, image_ir_mask_cut)
            loss1_mask_cut_vis = F.l1_loss(outputs_vis_mask_cut, image_vis_mask_cut)
            loss1 = loss1_back_vis + loss1_back_ir +  loss1_mask_cut_ir + loss1_mask_cut_vis

            loss2_back = F.l1_loss(gradient(outputs), torch.max(gradient(image_ir), gradient(vis_y_image)))
            loss2_vis_mask_cut = F.l1_loss(gradient(outputs_vis_mask_cut), gradient(image_vis_mask_cut))
            loss2_ir_mask_cut = F.l1_loss(gradient(outputs_ir_mask_cut), gradient(image_ir_mask_cut))
            loss2 =  loss2_back + loss2_vis_mask_cut + loss2_ir_mask_cut

            loss3_ir = ssim_loss(image_ir, outputs)
            loss3_vi = ssim_loss(vis_y_image, outputs)
            loss3_ir_mask = ssim_loss(image_ir_mask_cut, outputs_ir_mask_cut)
            loss3_vi_mask = ssim_loss(image_vis_mask_cut, outputs_vis_mask_cut)
            loss3 = (1 -loss3_ir) + (1-loss3_vi) + (1-loss3_ir_mask) + (1-loss3_vi_mask)

            loss = loss1 + loss2 + loss3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        fusionmodel.eval()
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        test_loss_all = 0
        with torch.no_grad():
            for image_vis_mask, vis_y_image, vis_cb_image, vis_cr_image, image_ir, image_ir_mask in test_tqdm:

                image_vis_mask = image_vis_mask.to(device)
                vis_y_image = vis_y_image.to(device)
                image_ir = image_ir.to(device)
                image_ir_mask = image_ir_mask.to(device)
                outputs = fusionmodel(image_vis_y=vis_y_image, image_ir=image_ir)

                outputs_vis_mask_cut = torch.mul(outputs, image_vis_mask)  # 得到融合图像，可见光掩膜中的图像
                outputs_ir_mask_cut = torch.mul(outputs, image_ir_mask)  # 得到融合图像，红外掩膜中的图像
                image_vis_mask_cut = torch.mul(vis_y_image, image_vis_mask)  # 得到可见光 Y通道，可见光掩膜中的图像
                image_ir_mask_cut = torch.mul(image_ir, image_ir_mask)  # 得到红外图像，红外掩膜中的图像

                loss1_back_vis = F.l1_loss(outputs, vis_y_image)
                loss1_back_ir = F.l1_loss(outputs, image_ir)
                loss1_mask_cut_ir = F.l1_loss(outputs_ir_mask_cut, image_ir_mask_cut)
                loss1_mask_cut_vis = F.l1_loss(outputs_vis_mask_cut, image_vis_mask_cut)
                loss1 = loss1_back_vis + loss1_back_ir + loss1_mask_cut_ir + loss1_mask_cut_vis

                loss2_back = F.l1_loss(gradient(outputs), torch.max(gradient(image_ir), gradient(vis_y_image)))
                loss2_vis_mask_cut = F.l1_loss(gradient(outputs_vis_mask_cut), gradient(image_vis_mask_cut))
                loss2_ir_mask_cut = F.l1_loss(gradient(outputs_ir_mask_cut), gradient(image_ir_mask_cut))
                loss2 = loss2_back + loss2_vis_mask_cut + loss2_ir_mask_cut

                loss3_ir = ssim_loss(image_ir, outputs)
                loss3_vi = ssim_loss(vis_y_image, outputs)
                loss3_ir_mask = ssim_loss(image_ir_mask_cut, outputs_ir_mask_cut)
                loss3_vi_mask = ssim_loss(image_vis_mask_cut, outputs_vis_mask_cut)
                loss3 = (1 - loss3_ir) + (1 - loss3_vi) + (1 - loss3_ir_mask) + (1 - loss3_vi_mask)

                loss = loss1 + loss2 + loss3
                test_loss_all = loss + test_loss_all

                print("loss1_back_vis", loss1_back_vis)
                print("loss1_back_ir", loss1_back_ir)
                print("loss1_ir_mask_cut_ir", loss1_mask_cut_ir)
                print("loss1_ir_mask_cut_vis", loss1_mask_cut_vis)

                print("loss2_max",loss2_back )
                print("loss2_vis_mask_cut", loss2_vis_mask_cut)
                print("loss2_ir_mask_cut",loss2_ir_mask_cut )

                print("loss3_ir_ssim",loss3_ir )
                print("loss3_vi_ssim",loss3_vi )
                print("loss3_ir_mask",loss3_ir_mask )
                print("loss3_vi_mask",loss3_vi_mask )

                print("loss1", loss1)
                print("loss2", loss2)
                print("loss3", loss3)
                print("all_loss",loss )
                print("------------------------------")
            print(test_loss_all)
        save_path = "Model/sigfusion/epoch_{"+str(i+1)+"}+loss="+str(test_loss_all.item())+".pth"
        torch.save(fusionmodel.state_dict(),  save_path)

if __name__ == "__main__":
    main()









