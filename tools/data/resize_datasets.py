from PIL import Image
import os
import os.path as osp
import random


dataset_root = "./datasets/transparent/Trans10K"
dataset_out_root = "/mnt/lustre/share_data/lixiangtai/datasets/transparent/Trans10K_cls12"


def check_dir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)


def random_size(shot_size):
    if shot_size > 1200:
        return random.randint(800, 1200)
    else:
        return shot_size


def resize_dataset(i_root, o_root, mode="train"):

    out_images_path = osp.join(o_root, mode, "images")
    out_masks_path = osp.join(o_root, mode, "masks_12")

    check_dir(o_root)
    check_dir(out_images_path)
    check_dir(out_masks_path)

    image_path = osp.join(i_root, mode, "images")
    mask_path = osp.join(i_root, mode, "masks_12")
    for i in os.listdir(image_path):
        img = Image.open(osp.join(image_path, i))
        basename, _ = os.path.splitext(i)
        mask = Image.open(osp.join(mask_path, basename+"_mask.png"))
        assert img.size == mask.size
        w, h = img.size

        if w > h:
            oh =random_size(h)
            ow = int(1.0 * w * oh / h)
        else:
            ow = random_size(w)

            oh = int(1.0 * h * ow / w)

        new_img = img.resize((ow, oh), Image.BILINEAR)
        new_mask = mask.resize((ow, oh), Image.NEAREST)
        new_img.save(osp.join(out_images_path, i))
        new_mask.save(osp.join(out_masks_path, basename+"_mask.png"))
        print("process image", i)


resize_dataset(dataset_root, dataset_out_root)
resize_dataset(dataset_root, dataset_out_root, mode='test')
resize_dataset(dataset_root, dataset_out_root, mode='validation')