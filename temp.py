import os
from PIL import Image

scene=['scene1','scene2','scene3','scene4']
path = '/home/farzad/Desktop/onGithub/CGAN/video/'

for sc in scene:
    players=os.listdir(path+sc+'/players/')
    for p in players:
        frames = os.listdir(path + sc + '/players/'+p)
        print(sc,p,len(frames))
        #for f in frames:
            #os.system('cp '+path + sc + '/players/'+p+'/'+f +' '+path+'4pix2pix/'+sc+'_'+p+'_'+f)



