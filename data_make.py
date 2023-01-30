import glob
import os

path = r'C:\Users\hbt13\Desktop\dataset02\img_Transvers\radiopaedia_org_covid-19-pneumonia-40_86625_0-dcm.gz'
path1 = 'test/unnormal'
i = 0
j = 0
filelist=os.listdir(path)
for i in filelist:
    # 判断该路径下的文件是否为图片
    if i.endswith('.png'):
        # 打开图片
        src = os.path.join(os.path.abspath(path), i)
        # 重命名
        dst = os.path.join(os.path.abspath(path1), '10_' + format(str(j), '0>3s') + '.png')
        # 执行操作
        os.rename(src, dst)
        j += 1
