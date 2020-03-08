import os 
import os.path as osp
import argparse
import tifffile as tif
import cv2 
import shutil

import os, zipfile
#打包目录为zip文件（未压缩）
def make_zip(source_dir, output_filename):
  zipf = zipfile.ZipFile(output_filename, 'w')
  pre_len = len(os.path.dirname(source_dir))
  for parent, dirnames, filenames in os.walk(source_dir):
    for filename in filenames:
      pathfile = os.path.join(parent, filename)
      arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径
      zipf.write(pathfile, arcname)
  zipf.close()
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="make sub")
    parser.add_argument("--img_dir", default="", help="", type=str)
    parser.add_argument("--save_dir", default="", help="", type=str)


    args = parser.parse_args()
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
        print('==> removing ',args.save_dir)
    os.makedirs(args.save_dir,exist_ok=True)

    for u in os.listdir(args.img_dir):
        subpath = osp.join(args.img_dir,u)
        img = cv2.imread(subpath,cv2.IMREAD_GRAYSCALE)
        tif.imwrite(osp.join(args.save_dir,u.replace('.png','_pred.tif')),img)
    print('==> make zip file')
    make_zip(args.save_dir,'submit.zip')
