import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import potrace




class Piture():
    def __init__(self,filepath):
        self.img=mpimg.imread(filepath)
        self.h,self.w,self.c=self.img.shape
        self.pre=np.ones(self.img.shape)
        self.gcode=['G28']
        self.x_max=150
        self.y_max=150
    #----------------------convert to gray scale----------------------------
    def gray_scale(self):
        print('RBG to gray scale...')
        gray = np.ones(self.img.shape) # new array for gray scale
        for i in range(self.h):
            for j in range(self.w):
                Y = (0.3*self.img[i,j,0]+0.59*self.img[i,j,1]+0.11*self.img[i,j,2])/255
                gray[i,j]=np.array([Y,Y,Y])
        self.pre=gray
        return gray
    #-----------------------------------------------------------------------

    #-----------------------prewiit edge detector---------------------------
    def prewiit(self):
        print('start to prewiit...')
        gray=self.pre
        result = np.zeros(self.img.shape) # new array for prewiit 
        for i in range(1,self.h-1):
            for j in range(1,self.w-1):
                Gx=-gray[i-1,j-1,0]-gray[i,j-1,0]-gray[i+1,j-1,0]+gray[i-1,j+1,0]+gray[i,j+1,0]+gray[i+1,j+1,0]
                Gy=-gray[i-1,j-1,0]-gray[i-1,j,0]-gray[i-1,j+1,0]+gray[i+1,j-1,0]+gray[i+1,j,0]+gray[i+1,j+1,0]
                G = (np.sqrt(Gx**2+Gy**2))
                if G>0.5:
                    G=1
                else:
                    G=0
                result[i,j]=np.array([G,G,G])
        self.pre=result
        # plt.imshow(self.pre)
        # plt.axis('off')
        # plt.show()
        return result
    #------------------------------------------------------------------------

    def denoise(self):
        print('start to denoise...')
        for i  in range(1,self.h):
            for j in range(1,self.w):
                if  self.pre[i,j,0]==1 and np.sum(self.pre[i-1:i+2,j-1:j+2,0])==1:
                    self.pre[i,j]=np.array([0,0,0])
        return self.pre

    def show(self):
        plt.imshow(self.pre)
        plt.axis('off')
        plt.show()

    def gen_gcode(self):
        print('generate gcode...')
        bmp=potrace.Bitmap(self.pre[:,:,0])
        path=bmp.trace()
        for curve in path:
            ratio=self.x_max/max(self.w,self.h) #normalize for drawing machine
            self.gcode.append('U') #抬筆
            self.gcode.append('G0 X%.4f Y%.4f'%(curve.start_point[0]*ratio,curve.start_point[1]*ratio)) #移動到起始點
            self.gcode.append('D') #下筆
            for segment in curve:
                if segment.is_corner:
                    self.gcode.append('G1 X%.4f Y%.4f'%(segment.c[0]*ratio,segment.c[1]*ratio)) #畫至corner的轉角點
                    self.gcode.append('G1 X%.4f Y%.4f'%(segment.end_point[0]*ratio,segment.end_point[1]*ratio)) #畫至corner的終點
                else:
                    self.gcode.append('G1 X%.4f Y%.4f'%(segment.end_point[0]*ratio,segment.end_point[1]*ratio)) #畫至Bezier segment的終點
        self.gcode.append('U') #抬筆
        return self.gcode
    
    def save_gcode(self):
        with open('output.txt','w') as f:
            for i in range(len(self.gcode)):
                f.write('%s\n'%self.gcode[i])

if __name__=='__main__':
    pic=Piture('img/bear.jpg') #輸入圖片的路徑
    pic.gray_scale()
    pic.prewiit()
    pic.denoise()
    gcode=pic.gen_gcode()
    pic.save_gcode()

