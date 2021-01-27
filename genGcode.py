import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageFilter
import numpy as np
import potrace
from scipy.ndimage.morphology import binary_closing
from scipy.ndimage.morphology import binary_fill_holes
import sys
sys.setrecursionlimit(1000000) # Increase recursive depth to run labeling

class Piture():
    def __init__(self,filepath):
        self.img = Image.open(filepath)
        self.img = np.array(self.img)
        print(self.img.shape)
        self.h,self.w,self.c=self.img.shape
        self.pre=np.ones(self.img.shape)
        self.gcode=['G28']
        self.connectPixel = 3 # connection expand max
        self.x_max=40
        self.y_max=40
      
    #----------------------convert to gray scale----------------------------
    def gray_scale(self):
        print('RBG to gray scale...')
        gray = np.ones(self.img.shape) # new array for gray scale
        for i in range(self.h):
            for j in range(self.w):
                Y = (0.3*self.img[i,j,0]+0.59*self.img[i,j,1]+0.11*self.img[i,j,2])/255
                if Y > 0.5: Y = 0
                else: Y = 1
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
        return result
    #------------------------------------------------------------------------

    #-----------------------Resize Picture (after grayScale)---------------------------
    def resizeAfterGrayScale(self, size):
        print('Resize to: ', size)
        tmp = self.pre[:, :, 0]
        tmp = Image.fromarray(np.uint8(tmp * 255), 'L')
        tmp = tmp.resize(size)
        # tmp.show()
        tmp = np.array(tmp)
        self.pre = np.zeros((tmp.shape[0], tmp.shape[1], 3))
        self.h,self.w = tmp.shape
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                G = 0
                if tmp[i, j] > 0: G = 1
                self.pre[i, j] = np.array([G, G, G])
    #------------------------------------------------------------------------

    #-----------------------Show the image on the screen---------------------------
    def show(self):
        plt.imshow(self.pre)
        plt.axis('off')
        plt.show()
    #------------------------------------------------------------------------

    #-----------------------Save the image---------------------------
    def saveImg(self, output):
        plt.imshow(self.pre)
        plt.axis('off')
        plt.imsave(output + '.jpg', self.pre)
        print('Save ' + output + '.jpg')
    #------------------------------------------------------------------------

    #-----------------------Generate Gcode---------------------------
    def gen_gcode(self):
        print('generate gcode...')
        # bmp=potrace.Bitmap(self.pre[:,:]) # binary fill
        bmp=potrace.Bitmap(self.pre[:,:,0])
        path=bmp.trace()
        flag = 0
        for curve in path:
            
            ratio=self.x_max/max(self.w,self.h) #normalize for drawing machine
            self.gcode.append('M280 P0 S60') #抬筆
            self.gcode.append('G0 X%.4f Y%.4f'%(curve.start_point[0]*ratio,curve.start_point[1]*ratio)) #移動到起始點
            self.gcode.append('M280 P0 S0') #下筆
            for segment in curve:
                if segment.is_corner:
                    self.gcode.append('G1 X%.4f Y%.4f'%(segment.c[0]*ratio,segment.c[1]*ratio)) #畫至corner的轉角點
                    self.gcode.append('G1 X%.4f Y%.4f'%(segment.end_point[0]*ratio,segment.end_point[1]*ratio)) #畫至corner的終點
                else:
                    self.gcode.append('G1 X%.4f Y%.4f'%(segment.end_point[0]*ratio,segment.end_point[1]*ratio)) #畫至Bezier segment的終點
        self.gcode.append('M280 P0 S60') #抬筆
        return self.gcode
    #------------------------------------------------------------------------
    
    #-----------------------Save Gcode---------------------------
    def save_gcode(self, output_name):
        with open(f'{output_name}_gcode.txt','w') as f:
            for i in range(len(self.gcode)):
                f.write('%s\n'%self.gcode[i])
    #------------------------------------------------------------------------
    

if __name__=='__main__':
    input_path = 'img/bear.jpg'
    output_name = 'out/' + input_path.split('/')[1].split('.')[0]

    pic=Piture(input_path)
    pic.gray_scale()
    pic.saveImg(f'{output_name}_grayscale')

   
    pic.saveImg(f'{output_name}_gcode')
    gcode = pic.gen_gcode()
    pic.save_gcode(output_name)
  