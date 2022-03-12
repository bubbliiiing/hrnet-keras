#--------------------------------------------#
#   该部分代码用于看网络结构
#--------------------------------------------#
from nets.hrnet import HRnet

if __name__ == "__main__":
    model = HRnet([480, 480, 3], 21, backbone='hrnetv2_w18')
    model.summary()

    for i,layer in enumerate(model.layers):
        print(i,layer.name)
