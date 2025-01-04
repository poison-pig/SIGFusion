
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from SIGFusion_dataset import SIGFusion_dataset
from utils import YCrCb2RGB,  clamp
from SIGFusion_model import SIGNet

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIGNet()
    model = model.to(device)
    model_path = "checkpoint/SIMFusion.pth"
    model.load_state_dict( torch.load(model_path, map_location=torch.device('cpu')) )
    model.eval()

    test_dataset = SIGFusion_dataset("test")
    print("the training dataset is length:{}".format(test_dataset.length))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    test_tqdm = tqdm(test_loader, total=len(test_loader))

    with torch.no_grad():
        for vis_y_image, cb, cr, ir_image,name in test_tqdm:

            vis_y_image = vis_y_image.to(device)
            cb = cb.to(device)
            cr = cr.to(device)
            ir_image = ir_image.to(device)

            fused_image = model(vis_y_image, ir_image)
            fused_image = clamp(fused_image)
            # print(fused_image.shape)

            rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
            print(rgb_fused_image.shape)
            rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)

            name = str(name)
            name = name.lstrip("('" )
            name = name.rstrip("',)")
            rgb_fused_image.save("Results/" + name )

if __name__ == "__main__":
    main()


