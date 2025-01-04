

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from PIL import Image
import glob
import os
from utils import RGB2YCrCb

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

to_tensor = transforms.Compose([transforms.ToTensor()])

class SIGFusion_dataset(Dataset):
    def __init__(self, split):
        super(SIGFusion_dataset, self).__init__()
        assert split in ['train', 'eval', 'test'], 'split must be "train"|"eval"|"test"'
        self.transform = to_tensor

        if split == 'train':
            data_dir_vis = 'Data/train/vi'
            data_dir_vis_mask = 'Data/train/Mask_vi'
            data_dir_ir = 'Data/train/ir'
            data_dir_ir_mask = 'Data/train/Mask_ir'

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_vis_mask, self.filenames_vis_mask = prepare_data_path(data_dir_vis_mask)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_ir_mask, self.filenames_ir_mask = prepare_data_path(data_dir_ir_mask)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'eval':
            data_dir_vis = 'Data/eval/vi'
            data_dir_vis_mask = 'Data/train/Mask_vi'
            data_dir_ir = 'Data/eval/ir'
            data_dir_ir_mask = 'Data/eval/Mask_ir'

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_vis_mask, self.filenames_vis_mask = prepare_data_path(data_dir_vis_mask)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.filepath_ir_mask, self.filenames_ir_mask = prepare_data_path(data_dir_ir_mask)

            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

        elif split == 'test':

            data_dir_vis = "Data/test/vi"
            data_dir_ir = "Data/test/ir"

            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            vis_path_mask = self.filepath_vis_mask[index]
            ir_path_mask = self.filepath_ir_mask[index]

            image_vis = self.transform(Image.open(vis_path).resize((512,512),resample=Image.BICUBIC))
            image_vis_mask = self.transform(Image.open(vis_path_mask).convert('L').resize((512,512),resample=Image.BICUBIC))
            image_ir = self.transform(Image.open(ir_path).convert('L').resize((512,512),resample=Image.BICUBIC))
            image_ir_mask = self.transform(Image.open(ir_path_mask).convert('L').resize((512,512),resample=Image.BICUBIC))
            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(image_vis)

            return image_vis_mask,vis_y_image, vis_cb_image, vis_cr_image,image_ir,image_ir_mask

        elif self.split == 'eval':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            vis_path_mask = self.filepath_vis_mask[index]
            ir_path_mask = self.filepath_ir_mask[index]

            image_vis = self.transform(Image.open(vis_path).resize((512, 512), resample=Image.BICUBIC))
            image_vis_mask = self.transform(Image.open(vis_path_mask).convert('L').resize((512, 512), resample=Image.BICUBIC))
            image_ir = self.transform(Image.open(ir_path).convert('L').resize((512, 512),resample=Image.BICUBIC))
            image_ir_mask = self.transform(Image.open(ir_path_mask).convert('L').resize((512, 512), resample=Image.BICUBIC))
            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(image_vis)

            return  image_vis_mask, vis_y_image, vis_cb_image, vis_cr_image, image_ir, image_ir_mask

        elif self.split=='test':
            vis_path = self.filepath_vis[index]
            print(int(index),vis_path)
            ir_path = self.filepath_ir[index]
            name = self.filenames_vis[index]

            image_vis = self.transform(Image.open(vis_path))
            image_ir = self.transform(Image.open(ir_path).convert('L'))
            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(image_vis)

            return  vis_y_image, vis_cb_image, vis_cr_image, image_ir,name

    def __len__(self):
        return self.length
