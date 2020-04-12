import torch
import pdb
if __name__ == '__main__':

    if 0:
        num_classes = 2
        model_fname1 = '/data/yyh/segmentation/dbseg/pretrain_models/DANet101.pth.tar'
        model_coco = torch.load(model_fname1)

        # weight
        model_coco["state_dict"]["head.conv6.1.weight"].resize_(num_classes,512,1,1)
        model_coco["state_dict"]["head.conv7.1.weight"].resize_(num_classes,512,1,1)
        model_coco["state_dict"]["head.conv8.1.weight"].resize_(num_classes,512,1,1)

        model_coco["state_dict"]["head.conv6.1.bias"].resize_(num_classes)
        model_coco["state_dict"]["head.conv7.1.bias"].resize_(num_classes)
        model_coco["state_dict"]["head.conv8.1.bias"].resize_(num_classes)

        torch.save(model_coco,model_fname1+'.{}'.format(num_classes))
    

    # if 1:
    #     num_classes = 2
    #     model_fname1 = '/data/yyh/segmentation/dbseg/cityscapes/scseocheadunet_model/cityscapes-scseocheadunet_101-b1024-c768-ms-poly/model_best.pth.tar'
    #     model_coco = torch.load(model_fname1)

    #     # weight
    #     model_coco["state_dict"]["dsn.4.weight"].resize_(num_classes,512,1,1)
    #     model_coco["state_dict"]["final_conv.weight"].resize_(num_classes,16,1,1)

    #     model_coco["state_dict"]["dsn.4.bias"].resize_(num_classes)
    #     model_coco["state_dict"]["final_conv.bias"].resize_(num_classes)

    #     torch.save(model_coco,model_fname1+'.{}'.format(num_classes))

    if 1:
        # for hrnet-ocr
        num_classes = 2
        model_fname1 = '/data/model_checkpoint_pytorch/hrnet_ocr_cs_8162_torch11.pth'
        model_coco = torch.load(model_fname1)

        # weight
        model_coco["model.aux_head.3.weight"].resize_(num_classes,720,1,1)
        model_coco["model.cls_head.weight"].resize_(num_classes,512,1,1)

        model_coco["model.aux_head.3.bias"].resize_(num_classes)
        model_coco["model.cls_head.bias"].resize_(num_classes)

        torch.save(model_coco,'{}_{}.pth'.format(model_fname1.split('.')[0],num_classes))
    
            
                    