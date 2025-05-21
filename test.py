from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
import os
import numpy as np

from net_fusion_restormer import Net
from utils.Evaluator import Evaluator
import torch
import torch.nn as nn
from utils.img_read_save import img_save, image_read_cv2, img_save_test
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# ckpt_path=r"models/CDDFuse_zhou_30_10-12-12-25.pth"
# for dataset_name in ["TNO","RoadScene"]:
for ssim_wight in [1]:
    for mse_wight in [1]:
        for grad_wight in [1]:

            ckpt_path = r"models/20-01-10.pth"

            if os.path.exists(ckpt_path):
                print(ckpt_path)
                for dataset_name in ['40_pairs_tno_vot']:
                    print("\n" * 2 + "=" * 80)
                    model_name = "CDDFuse    "
                    print("The test result of " + dataset_name + ' :')
                    test_folder = os.path.join('test_img', dataset_name)
                    test_out_folder = os.path.join('test_result_', dataset_name)

                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    net = Net().to(device)

                    net.load_state_dict(torch.load(ckpt_path)['net'])

                    net.eval()

                    with torch.no_grad():
                        for img_name in os.listdir(os.path.join(test_folder, "ir")):
                            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                                          np.newaxis, np.newaxis, ...] / 255.0
                            data_VIS = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='GRAY')[
                                           np.newaxis, np.newaxis, ...] / 255.0

                            data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
                            data_VIS, data_IR = data_VIS.cuda(), data_IR.cuda()

                            data_Fuse = net(data_VIS, data_IR)
                            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (
                                        torch.max(data_Fuse) - torch.min(data_Fuse))
                            fi = np.squeeze((data_Fuse * 255).cpu().numpy())
                            img_save_test(fi, img_name, test_out_folder)

                    eval_folder = test_out_folder
                    ori_img_folder = test_folder

                    metric_result = np.zeros((8))
                    for img_name in os.listdir(os.path.join(ori_img_folder, "ir")):
                        ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), 'GRAY')
                        vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), 'GRAY')
                        fi = image_read_cv2(os.path.join(eval_folder, img_name), 'GRAY')
                        metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi)
                                                      , Evaluator.SF(fi), Evaluator.MI(fi, ir, vi)
                                                      , Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi)
                                                      , Evaluator.Qabf(fi, ir, vi), Evaluator.SSIM(fi, ir, vi)])

                    metric_result /= len(os.listdir(eval_folder))
                    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
                    print(model_name + '\t' + str(np.round(metric_result[0], 2)) + '\t'
                          + str(np.round(metric_result[1], 2)) + '\t'
                          + str(np.round(metric_result[2], 2)) + '\t'
                          + str(np.round(metric_result[3], 2)) + '\t'
                          + str(np.round(metric_result[4], 2)) + '\t'
                          + str(np.round(metric_result[5], 2)) + '\t'
                          + str(np.round(metric_result[6], 2)) + '\t'
                          + str(np.round(metric_result[7], 2))
                          )
                    print("=" * 80)
