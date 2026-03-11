import torch
import os
from numpy import *
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms

from 机器学习 import ori_path
from 机器学习.model import resnet152


def img_predict(img_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet152()

    model_weight_path = ori_path + "./resnet152-b121ed2d.pth"   # 加载resnet的预训练模型
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)

    model.load_state_dict(torch.load('./机器学习/wheatDiseaseDict.pth'))
    model.to(device)
    class_names = ['Crown and Root Rot', 'Healthy Wheat', 'Leaf Rust', 'Wheat Loose Smut']
    img_path = str(img_path)
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    image = Image.open(img_path)
    plt.imshow(image)
    trans = transforms.Compose([transforms.Resize((180, 180)), transforms.ToTensor()])
    # 将图片切换为RGB格式与进行转换操作
    image = image.convert("RGB")
    image = trans(image)
    # 将图片维度扩展一维,得到4维图片
    image = torch.unsqueeze(image, dim=0)
    # 开始验证
    model.eval()  # 关闭梯度，将模型调整为测试模式
    with torch.no_grad():
        output = torch.squeeze(model(image.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).cpu().numpy()
        output = model(image.to(device))
        predict_cla = output.argmax(1).clone().detach().item()
    ImgIdentResult = class_names[predict_cla]
    PredictAccuracy = []
    PredictAccuracy.append(round(predict[predict_cla].cpu().tolist() * 100, 2))
    PredictAccuracy = array(PredictAccuracy).tolist()
    return ImgIdentResult, PredictAccuracy

# # 创建一个与训练时类名顺序一致的列表
# class_names = ['healthy', 'leaf_rust', 'stem_rust']
# # 载入模型：
# model = torch.load('xiaomai.pth')
# model.to(device)
# image = Image.open(r'D:\Dataset\Rice Leaf Disease Images\train\Blast\BLAST1_121.JPG')
# trans = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
# image = image.convert("RGB")
# image = trans(image)
# image = torch.unsqueeze(image, dim=0)

# # 开始验证
# model.eval()             # 关闭梯度，将模型调整为测试模式
# with torch.no_grad():    # 梯度清零
#     outputs = model(image.to(device))
#     # ans = torch.tensor(outputs.argmax(1)).item()
#     ans = outputs.argmax(1).clone().detach().item()  # 最大的值即为预测结果，找出最大值在数组中的序号
#     print(class_names[ans])                          # 输出的是那种即为预测结果
