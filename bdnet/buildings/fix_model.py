import torch
import pdb
if __name__ == '__main__':

    if 1:
        num_classes = 2
        # model_fname1 = '/data/yao/detections/pretrained_models/htc_r50_fpn_20e_20190408-c03b7015.pth'
        model_fname1 = '/data/yyh/segmentation/dbseg/pretrain_models/DANet101.pth.tar'
        # model_fname1 = '/data/model_checkpoint_pytorch/coco_casdhx10164_dcn_ms_12epo_4830.pth'
        model_coco = torch.load(model_fname1)

        # weight
        model_coco["state_dict"]["head.conv6.1.weight"].resize_(num_classes,512,1,1)
        model_coco["state_dict"]["head.conv7.1.weight"].resize_(num_classes,512,1,1)
        model_coco["state_dict"]["head.conv8.1.weight"].resize_(num_classes,512,1,1)

        model_coco["state_dict"]["head.conv6.1.bias"].resize_(num_classes)
        model_coco["state_dict"]["head.conv7.1.bias"].resize_(num_classes)
        model_coco["state_dict"]["head.conv8.1.bias"].resize_(num_classes)

        torch.save(model_coco,model_fname1+'.{}'.format(num_classes))
    
model_coco["state_dict"]["head.conv6.1.weight"].resize_(num_classes,512,1,1)
model_coco["state_dict"]["head.conv7.1.weight"].resize_(num_classes,512,1,1)
model_coco["state_dict"]["head.conv8.1.weight"].resize_(num_classes,512,1,1)

model_coco["state_dict"]["head.conv6.1.bias"].resize_(num_classes)
model_coco["state_dict"]["head.conv7.1.bias"].resize_(num_classes)
model_coco["state_dict"]["head.conv8.1.bias"].resize_(num_classes)

torch.save(model_coco,model_fname1+'.{}'.format(num_classes))
            
                    