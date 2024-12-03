import os
import random

from PIL import Image

# Define the folder paths
folder1 = '/home/farzad/Desktop/onGithub/CGAN/video/player33/concatenated'
files=list(os.listdir(folder1))
files.remove('train')
files.remove('test')
files.remove('val')
fi=len(files)
train_num=int(.7*fi)
for i in range(0,train_num):
    l = len(files)
    r=random.randint(0,l-1)
    os.system('cp '+folder1+'/'+files[r]+' '+folder1+'/train/')
    files.remove(files[r])

l=len(files)
test_num=int(.2*l)
for i in range(0,test_num):
    l = len(files)
    r=random.randint(0,l-1)
    os.system('cp '+folder1+'/'+files[r]+' '+folder1+'/test/')
    files.remove(files[r])

for f in files:
    os.system('cp '+folder1+'/'+f+' '+folder1+'/val/')
