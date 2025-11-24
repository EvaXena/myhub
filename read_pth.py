import torch
import pprint

##===================================================
###可以使用
#####检查变量文件类型：isinstance(variable,dtype)
#####字典的每个元素调用：.items()
#####按键名调用字典内部元素:dict.get('keyname')
##===================================================
###用于记录
#####pytorch1.9的.pth文件类型是dict


torch_file_path = '../checkpoint.pth'


def show_all_keyname(dict):
    for key,_ in dict.items():
            print('key_name:',key)


def read_torch_file():
    model = torch.load(torch_file_path)
    bool = isinstance(model,dict)
    pprint.pprint(bool)
    return model
    
def read_key(dict,keyname):
    data = dict.get(keyname)
    pprint.pprint(data)

#文件读取
# for root,dirs,files in os.walk(path):
#     for file in files:
#         if file.endswith(('.jpg','.png'))


if __name__ == '__main__':
    model = read_torch_file()
    read_key(model,'cfg')
