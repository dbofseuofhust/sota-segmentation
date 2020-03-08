import os
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
from PIL import Image
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import torch
import torch.utils.data as data
import albumentations as albu
import torchvision.transforms as transforms
import cv2
import numbers
import shutil
from torchvision.transforms import functional as F
import scipy.io as scio

import encoding.utils as utils
from encoding.nn import SegmentationLosses
from encoding.nn import SegmentationMultiLosses
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import test_batchify_fn
from encoding.models import get_model, get_segmentation_model, MultiEvalModule
from tqdm import tqdm

import argparse

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch \
            Segmentation')
        # model and dataset 
        parser.add_argument('--model', type=str, default='encnet',
                            help='model name (default: encnet)')
        parser.add_argument('--backbone', type=str, default='resnet50',
                            help='backbone name (default: resnet50)')
        parser.add_argument('--dataset', type=str, default='cityscapes',
                            help='dataset name (default: pascal12)')
        parser.add_argument('--data-folder', type=str,
                            default=os.path.join(os.environ['HOME'], 'data'),
                            help='training dataset folder (default: \
                            $(HOME)/data)')
        parser.add_argument('--workers', type=int, default=16,
                            metavar='N', help='dataloader threads')
        parser.add_argument('--base-size', type=int, default=608,
                            help='base image size')
        parser.add_argument('--crop-size', type=int, default=576,
                            help='crop image size')
        # training hyper params

        parser.add_argument('--aux', action='store_true', default= False,
                            help='Auxilary Loss')
        parser.add_argument('--se-loss', action='store_true', default= False,
                            help='Semantic Encoding Loss SE-loss')
        parser.add_argument('--epochs', type=int, default=None, metavar='N',
                            help='number of epochs to train (default: auto)')
        parser.add_argument('--start_epoch', type=int, default=0,
                            metavar='N', help='start epochs (default:0)')
        parser.add_argument('--batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            training (default: auto)')
        parser.add_argument('--test-batch-size', type=int, default=None,
                            metavar='N', help='input batch size for \
                            testing (default: same as batch size)')

        # warmup params
        parser.add_argument('--warmup', type=bool, default=False,
                            help='whether to use warmup.')
        parser.add_argument('--warmup-epoch', type=int, default=10,
                            help='warmup epochs.')
        parser.add_argument('--mutil-steps', type=str, default='80,140',
                            help='learning rate decay steps.')
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='gamma')
        parser.add_argument('--warmup-factor', type=float, default=0.1,
                            help='warmup factor.')
        parser.add_argument('--warmup-method', type=str, default='linear',
                            help='choose from (linear or constant).')

        parser.add_argument('--is-dilated', type=bool, default=False,
                            help='whether to use dilated conv.')

        # optimizer params
        parser.add_argument('--lr', type=float, default=None, metavar='LR',
                            help='learning rate (default: auto)')
        parser.add_argument('--lr-scheduler', type=str, default='poly',
                            help='learning rate scheduler (default: poly)')
        parser.add_argument('--lr-step', type=int, default=None,
                            help='lr step to change lr')
        parser.add_argument('--momentum', type=float, default=0.9,
                            metavar='M', help='momentum (default: 0.9)')
        parser.add_argument('--weight-decay', type=float, default=1e-4,
                            metavar='M', help='w-decay (default: 1e-4)')
        # cuda, seed and logging
        parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        parser.add_argument('--seed', type=int, default=1, metavar='S',
                            help='random seed (default: 1)')
        parser.add_argument('--log-root', type=str,
                            default='./cityscapes/log', help='set a log path folder')

        # checking point
        parser.add_argument('--resume', type=str, default=None,
                            help='put the path to resuming file if needed')
        parser.add_argument('--resume-dir', type=str, default=None,
                            help='put the path to resuming dir if needed')
        parser.add_argument('--checkname', type=str, default='default',
                            help='set the checkpoint name')
        parser.add_argument('--model-zoo', type=str, default=None,
                            help='evaluating on model zoo model')
        # finetuning pre-trained models
        parser.add_argument('--ft', action='store_true', default= False,
                            help='finetuning on a different dataset')
        parser.add_argument('--ft-resume', type=str, default=None,
                            help='put the path of trained model to finetune if needed ')
        parser.add_argument('--pre-class', type=int, default=None,
                            help='num of pre-trained classes \
                            (default: None)')

        # evaluation option
        parser.add_argument('--ema', action='store_true', default= False,
                            help='using EMA evaluation')
        parser.add_argument('--eval', action='store_true', default= False,
                            help='evaluating mIoU')
        parser.add_argument('--no-val', action='store_true', default= False,
                            help='skip validation during training')
        # test option
        parser.add_argument('--test-folder', type=str, default=None,
                            help='path to test image folder')
        parser.add_argument('--multi-scales',action="store_true", default=False,
                            help="testing scale,default:1.0(single scale)")
        # multi grid dilation option
        parser.add_argument("--multi-grid", action="store_true", default=False,
                            help="use multi grid dilation policy")
        parser.add_argument('--multi-dilation', nargs='+', type=int, default=None,
                            help="multi grid dilation list")
        parser.add_argument('--scale', action='store_false', default=True,
                           help='choose to use random scale transform(0.75-2),default:multi scale')
        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        # default settings for epochs, batch_size and lr
        if args.epochs is None:
            epoches = {
                'pascal_voc': 50,
                'pascal_aug': 50,
                'pcontext': 80,
                'ade20k': 160,
                'cityscapes': 180,
            }
            args.epochs = epoches[args.dataset.lower()]
        if args.batch_size is None:
            args.batch_size = 4 * torch.cuda.device_count()
        if args.test_batch_size is None:
            args.test_batch_size = args.batch_size
        if args.lr is None:
            lrs = {
                'pascal_voc': 0.0001,
                'pascal_aug': 0.001,
                'pcontext': 0.001,
                'ade20k': 0.01,
                'cityscapes': 0.01,
            }
            args.lr = lrs[args.dataset.lower()] / 8 * args.batch_size
        return args

torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable

class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None, 
                 target_transform=None, base_size=520, crop_size=480,
                 logger=None, scale=True):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        self.logger = logger
        self.scale = scale

        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))

        if not self.scale:
            if self.logger is not None:
                self.logger.info('single scale training!!!')

        self.min_size,self.max_size = 1024,2048

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def cal_new_size(self, im_h, im_w, min_size, max_size):
        if im_h < im_w:
            if im_h < min_size:
                ratio = 1.0 * min_size / im_h
                im_h = min_size
                im_w = round(im_w * ratio)
            elif im_h > max_size:
                ratio = 1.0 * max_size / im_h
                im_h = max_size
                im_w = round(im_w * ratio)
            else:
                ratio = 1.0
        else:
            if im_w < min_size:
                ratio = 1.0 * min_size / im_w
                im_w = min_size
                im_h = round(im_h * ratio)
            elif im_w > max_size:
                ratio = 1.0 * max_size / im_w
                im_w = max_size
                im_h = round(im_h * ratio)
            else:
                ratio = 1.0
        return im_h, im_w, ratio

    def _val_sync_transform(self, img, mask):
        assert self.base_size > self.crop_size

        # to keep the same scale
        w, h = img.size

        if (min(w,h) > self.base_size) or (min(w,h)<self.base_size and max(w,h)>self.base_size):
            ow, oh = w, h
            new_w, new_h = (ow // 32+1)*32, (oh // 32+1)*32
            padh = new_h - oh
            padw = new_w - ow
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)
        else:
            ow, oh = self.crop_size, self.crop_size
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)

        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        assert self.base_size > self.crop_size
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # random hflip
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # random rotate90
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)

        # random rotate180
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_180)
            mask = mask.transpose(Image.ROTATE_180)

        # random rotate270
        if random.random() < 0.5:
            img = img.transpose(Image.ROTATE_270)
            mask = mask.transpose(Image.ROTATE_270)

        # [todo] colorjitter
        if random.random() < 0.5:
            cj = ColorJitter(brightness=0.5,contrast=0.5,saturation=0,hue=0.5)
            img = cj(img)

        # for img size < base_size, pad it into crop_size
        w, h = img.size
        # for large img, random crop
        if min(w,h) > self.base_size:
            # random crop
            x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
            y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # for mid img, pad it and random crop
        elif min(w,h)<self.base_size and max(w,h)>self.base_size:
            padh = self.crop_size - h if h < self.crop_size else 0
            padw = self.crop_size - w if w < self.crop_size else 0

            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=255)

            # random crop
            x1 = random.randint(0, w - self.crop_size) if w > self.crop_size else 0
            y1 = random.randint(0, h - self.crop_size) if h > self.crop_size else 0
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
            mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # for small img, resize it into the same size
        else:
            # resize
            ow, oh = self.crop_size, self.crop_size
            img = img.resize((ow, oh), Image.BILINEAR)
            mask = mask.resize((ow, oh), Image.NEAREST)

        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()

class MonusacSegmentation(BaseDataset):
    BASE_DIR = 'monusac'
    NUM_CLASS = 5

    def __init__(self, root='/data/Dataset/Testing images', split='train',
                 mode=None, transform=None, target_transform=None, base_size=768, crop_size=1024, **kwargs):
        super(MonusacSegmentation, self).__init__(
            root, split, mode, transform, target_transform, base_size=base_size, crop_size=crop_size, **kwargs)
        # assert exists
        assert os.path.exists(root), "Please download the dataset!!"

        self.images = []
        for u in os.listdir(root):
            for v in os.listdir(os.path.join(root,u)):
                if v.endswith('tif'):
                    self.images.append(os.path.join(root,u,v))

        if len(self.images) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        # temp = '{}.jpg'.format(self.images[index].split('.')[0])
        # shutil.copyfile(self.images[index],temp)
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'vis':
            if self.transform is not None:
                # to keep the same scale
                img = self.transform(img)
            return img, '{}*{}'.format(self.root,self.images[index])

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        return torch.from_numpy(target).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0

def test(args):
    # output folder
    outdir = '%s/danet_vis' % (args.dataset)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # dataset
    testset = MonusacSegmentation(root=args.test_folder,split='test', mode='vis',
                                           transform=input_transform)
    # dataloader
    loader_kwargs = {'num_workers': args.workers, 'pin_memory': True} \
        if args.cuda else {}
    test_data = data.DataLoader(testset, batch_size=args.test_batch_size,
                                drop_last=False, shuffle=False,
                                collate_fn=test_batchify_fn, **loader_kwargs)
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, aux=args.aux,
                                       se_loss=args.se_loss, norm_layer=nn.BatchNorm2d,
                                       base_size=args.base_size, crop_size=args.crop_size,
                                       multi_grid=args.multi_grid, multi_dilation=args.multi_dilation)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'], strict=False)

    print(model)
    num_class = testset.num_class
    evaluator = MultiEvalModule(model, testset.num_class, multi_scales=args.multi_scales).cuda()
    evaluator.eval()

    tbar = tqdm(test_data)

    def eval_batch(image, dst, evaluator, eval_mode):
        if eval_mode:
            # evaluation mode on validation set
            targets = dst
            outputs = evaluator.parallel_forward(image)

            batch_inter, batch_union, batch_correct, batch_label = 0, 0, 0, 0
            for output, target in zip(outputs, targets):
                correct, labeled = utils.batch_pix_accuracy(output.data.cpu(), target)
                inter, union = utils.batch_intersection_union(
                    output.data.cpu(), target, testset.num_class)
                batch_correct += correct
                batch_label += labeled
                batch_inter += inter
                batch_union += union
            return batch_correct, batch_label, batch_inter, batch_union
        else:
            # Visualize and dump the results
            im_paths = dst
            outputs = evaluator.parallel_forward(image)
            predicts = [torch.max(output, 1)[1].cpu().numpy() + testset.pred_offset
                        for output in outputs]
            for predict, impath in zip(predicts, im_paths):
                # for vis 
                # mask = utils.get_mask_pallete(predict, args.dataset)

                # for submit
                # mask = Image.fromarray(predict.squeeze().astype('uint8'))

                # for norm outputs
                _root, _impath = impath.split('*')
                impath = _impath.replace(_root+'/','')
                outname = os.path.splitext(impath)[0] + '.png'
                mask = predict.squeeze().astype('uint8')
                norm_outputs(mask,outdir,outname)

                # scale into original shape
                # outname = os.path.splitext(impath)[0] + '.png'
                # mask.save(os.path.join(outdir, outname))
            # dummy outputs for compatible with eval mode
            return 0, 0, 0, 0

    total_inter, total_union, total_correct, total_label = \
        np.int64(0), np.int64(0), np.int64(0), np.int64(0)
    for i, (image, dst) in enumerate(tbar):
        if torch_ver == "0.3":
            image = Variable(image, volatile=True)
            correct, labeled, inter, union = eval_batch(image, dst, evaluator, args.eval)
        else:
            with torch.no_grad():
                correct, labeled, inter, union = eval_batch(image, dst, evaluator, args.eval)
        pixAcc, mIoU, IoU = 0, 0, 0
        if args.eval:
            total_correct += correct.astype('int64')
            total_label += labeled.astype('int64')
            total_inter += inter.astype('int64')
            total_union += union.astype('int64')
            pixAcc = np.float64(1.0) * total_correct / (np.spacing(1, dtype=np.float64) + total_label)
            IoU = np.float64(1.0) * total_inter / (np.spacing(1, dtype=np.float64) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
    return pixAcc, mIoU, IoU, num_class

def eval_multi_models(args):
    if args.resume_dir is None or not os.path.isdir(args.resume_dir):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume_dir))
    for resume_file in os.listdir(args.resume_dir):
        if resume_file == 'model_best.pth.tar':
            args.resume = os.path.join(args.resume_dir, resume_file)
            print(args.resume)
            assert os.path.exists(args.resume)
            if not args.eval:
                test(args)
                continue
            pixAcc, mIoU, IoU, num_class = test(args)

            txtfile = args.resume
            txtfile = txtfile.replace('pth.tar', 'txt')
            if not args.multi_scales:
                txtfile = txtfile.replace('.txt', 'result_mIoU_%.4f.txt' % mIoU)
            else:
                txtfile = txtfile.replace('.txt', 'multi_scale_result_mIoU_%.4f.txt' % mIoU)
            fh = open(txtfile, 'w')
            print("================ Summary IOU ================\n")
            for i in range(0, num_class):
                print("%3d: %.4f\n" % (i, IoU[i]))
                fh.write("%3d: %.4f\n" % (i, IoU[i]))
            print("Mean IoU over %d classes: %.4f\n" % (num_class, mIoU))
            print("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
            fh.write("Mean IoU over %d classes: %.4f\n" % (num_class, mIoU))
            fh.write("Pixel-wise Accuracy: %2.2f%%\n" % (pixAcc * 100))
            fh.close()
    print('Evaluation is finished!!!')

def norm_outputs(mask,save_dir,name):
    num_classes = 4
    cell_classes = {1:'Epithelial',2:'Lymphocyte',3:'Neutrophil',4:'Macrophage'}
    for i in range(num_classes):
        submask = np.zeros(mask.shape)
        submask[mask == (i+1)] = (i+1)
        up,down = name.split('/')
        subsave = os.path.join(save_dir,up,down.split('.')[0],cell_classes[i+1])
        os.makedirs(subsave,exist_ok=True)
        scio.savemat(os.path.join(subsave,'{}_mask.mat'.format(i+1)),{'n_ary_mask':submask})

if __name__ == "__main__":

    args = Options().parse()
    args.model = 'hcocheadunet'
    args.resume_dir = 'monusac/hcocheadunet_model/exp8-hcocheadunet_resnet101-warmup10-lr002-seprs-bsize1024-csize896-cj-allrt'
    args.base_size = 1024
    args.crop_size = 896
    args.workers = 4
    args.backbone = 'resnet101'
    args.log_root
    args.dataset = 'monusac'
    args.multi_dilation = [4,8,16]
    args.multi_grid
    args.test_folder = '/data/Dataset/Testing images' # modify here for your test images dir

    torch.manual_seed(args.seed)
    args.test_batch_size = torch.cuda.device_count()
    eval_multi_models(args)
