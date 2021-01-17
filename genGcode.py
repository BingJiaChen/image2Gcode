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

    #-----------------------Smooth the edge---------------------------
    def smooth(self):
        print("start to smooth")
        tmp = self.pre[:, :, 0]
        tmp = Image.fromarray(np.uint8(tmp * 255), 'L')
        tmp = tmp.filter(ImageFilter.SMOOTH)
        tmp = np.array(tmp)
        self.pre = np.zeros((tmp.shape[0], tmp.shape[1], 3))
        self.h,self.w = tmp.shape
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                G = 0
                if tmp[i, j] > 0: G = 1
                self.pre[i, j] = np.array([G, G, G])
    #------------------------------------------------------------------------

    #-----------------------Sharpen the edge---------------------------
    def sharpen(self):
        print("start to sharpen")
        tmp = self.pre[:, :, 0]
        tmp = Image.fromarray(np.uint8(tmp * 255), 'L')
        tmp = tmp.filter(ImageFilter.SHARPEN)
        tmp = np.array(tmp)
        self.pre = np.zeros((tmp.shape[0], tmp.shape[1], 3))
        self.h,self.w = tmp.shape
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                G = 0
                if tmp[i, j] > 0: G = 1
                self.pre[i, j] = np.array([G, G, G])
    #------------------------------------------------------------------------

    def denoise(self):
        print('start to denoise...')
        for i  in range(1,self.h):
            for j in range(1,self.w):
                if  self.pre[i,j,0]==1 and np.sum(self.pre[i-1:i+2,j-1:j+2,0])==1:
                    self.pre[i,j]=np.array([0,0,0])
        return self.pre        

    #-----------------------Edge Thinning (make edge to 1 pixel)---------------------------
    def edge_thinning(self):
        print('start edge thinning...')
        deletable = np.zeros(self.pre.shape)
        thin_times = 2
        for i in range(thin_times):
            for i in range(1,self.h-1):
                for j in range(1,self.w-1):
                    if self.pre[i,j,0]==0:
                        continue
                    sk = self.pre[i-1:i+2,j-1:j+2,0]
                    sk90 = np.rot90(sk)
                    sk180 = np.rot90(sk,2)
                    sk270 = np.rot90(sk,3)
                    sk1 = np.array([[0,0,0],[2,1,2],[1,1,1]])
                    sk2 = np.array([[2,0,0],[1,1,0],[2,1,2]])
                    if np.sum(sk==sk1)==7 or np.sum(sk90==sk1)==7 or np.sum(sk180==sk1)==7 or np.sum(sk270==sk1)==7:
                        deletable[i,j]=np.array([1,1,1])
                    if np.sum(sk==sk2)==6 or np.sum(sk90==sk2)==6 or np.sum(sk180==sk2)==6 or np.sum(sk270==sk2)==6:
                        deletable[i,j]=np.array([1,1,1])
            if np.sum(deletable)==0:
                break
            print("deleting ",np.sum(deletable)/3,"pixels")
            self.pre=self.pre-deletable
            deletable = np.zeros(self.pre.shape)
        return self.pre
    #------------------------------------------------------------------------

    #-----------------------Connect broken line---------------------------
    def connect(self):
        print('start to connect...')
        addable = np.zeros(self.pre.shape)
        for times in range(self.connectPixel):
            for i in range(1,self.h-1):
                for j in range(1,self.w-1):
                    sk = self.pre[i-1:i+2,j-1:j+2,0]
                    if np.sum(sk)<=1:
                        continue
                    sk90 = np.rot90(sk)
                    sk180 = np.rot90(sk,2)
                    sk270 = np.rot90(sk,3)
                    sk1 = np.array([[2,0,1],[0,0,0],[1,0,2]])
                    sk2 = np.array([[0,1,0],[2,0,2],[0,1,0]])
                    sk3 = np.array([[0,1,0],[0,0,0],[1,2,0]])
                    sk4 = np.array([[0,1,0],[0,0,0],[0,2,1]])
                    sk5 = np.array([[0,0,0],[0,1,0],[0,1,0]])
                    sk6 = np.array([[0,0,0],[0,1,0],[1,0,0]])
                    if np.sum(sk==sk1)==7 or np.sum(sk90==sk1)==7:
                        addable[i,j]=np.array([1,1,1])
                    if np.sum(sk==sk2)==7 or np.sum(sk90==sk2)==7:
                        addable[i,j]=np.array([1,1,1])
                    if np.sum(sk==sk3)==8 or np.sum(sk90==sk3)==8 or np.sum(sk180==sk3)==8 or np.sum(sk270==sk3)==8:
                        addable[i,j]=np.array([1,1,1])
                    if np.sum(sk==sk4)==8 or np.sum(sk90==sk4)==8 or np.sum(sk180==sk4)==8 or np.sum(sk270==sk4)==8:
                        addable[i,j]=np.array([1,1,1])
                    if np.sum(sk==sk5)==9:
                        addable[i-1,j]=np.array([1,1,1])
                    elif np.sum(sk==np.rot90(sk5))==9:
                        addable[i,j-1]=np.array([1,1,1])
                    elif np.sum(sk==np.rot90(sk5,2))==9:
                        addable[i+1,j]=np.array([1,1,1])
                    elif np.sum(sk==np.rot90(sk5,3))==9:
                        addable[i,j+1]=np.array([1,1,1])
                    if np.sum(sk==sk6)==9:
                        addable[i-1,j+1]=np.array([1,1,1])
                    elif np.sum(sk==np.rot90(sk6))==9:
                        addable[i-1,j-1]=np.array([1,1,1])
                    elif np.sum(sk==np.rot90(sk5,2))==9:
                        addable[i+1,j-1]=np.array([1,1,1])
                    elif np.sum(sk==np.rot90(sk5,3))==9:
                        addable[i+1,j+1]=np.array([1,1,1])
            self.pre = self.pre+addable
            print("connecting for",times+1,"times and connect",np.sum(addable)/3,"pixels")
            if np.sum(addable)==0:
                break
            addable = np.zeros(self.pre.shape)
        return self.pre
    #------------------------------------------------------------------------

    #-----------------------Prun outsided line---------------------------
    def pruning(self):
        print('start pruning...')
        deletable = np.zeros(self.pre.shape)
        for times in range(self.connectPixel):
            for i in range(1,self.h-1):
                for j in range(1,self.w-1):
                    if self.pre[i,j,0]==0:
                        continue
                    sk = self.pre[i-1:i+2,j-1:j+2,0]
                    sk1 = np.array([[0,0,0],[0,1,0],[0,1,0]])
                    sk2 = np.array([[0,0,0],[0,1,0],[1,0,0]])
                    if np.sum(sk==sk1)==9 or np.sum(sk==np.rot90(sk1))==9 or np.sum(sk==np.rot90(sk1,2))==9 or np.sum(sk==np.rot90(sk1,3))==9:
                        deletable[i,j]=np.array([1,1,1])
                    if np.sum(sk==sk2)==9 or np.sum(sk==np.rot90(sk2))==9 or np.sum(sk==np.rot90(sk2,2))==9 or np.sum(sk==np.rot90(sk2,3))==9:
                        deletable[i,j]=np.array([1,1,1])
            self.pre=self.pre-deletable
            print('pruning for',times + 1,'times , deleting',np.sum(deletable)/3,"pixels")
            if np.sum(deletable)==0:
                break
            deletable = np.zeros(self.pre.shape)
        return self.pre
    #------------------------------------------------------------------------

    #-----------------------Component Labeling---------------------------
    def dfs(self, arr, x, y, label):
        if x < 0 or x >= arr.shape[0]: return
        if y < 0 or y >= arr.shape[1]: return
        if arr[x, y] != -1: return
        arr[x, y] = label
        for i in range(x - 1, x + 2): # x - 1 ~ x + 1
            for j in range(y - 1, y + 2): # y - 1 ~ y + 1
                self.dfs(arr, i, j, label)

    def labeling(self):
        print('Start labeling')
        tmp = np.array(self.pre[:, :, 0]) # Only need 2d
        print(tmp.shape)
        label = 1

        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                tmp[i, j] = int(tmp[i, j])
                if tmp[i, j] == 1: tmp[i, j] = -1 # -1 means unlabel white part
        
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                if tmp[i, j] == -1:
                    self.dfs(tmp, i, j, label)
                    label += 1 # Next Component will label + 1
        print('Component Count: {}'.format(label))

        # Color mapping
        colorMap = {}
        for i in range(1, label + 1): # 1 ~ label
            colorMap[i] = np.random.randint(255, size=3)

        # Print Different color line
        pic = np.zeros(self.pre.shape)
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                if tmp[i, j] != 0:
                    pic[i, j] = colorMap[tmp[i, j]] / 255 # Float number shoud be in [0 ~ 1]
                
        plt.imshow(pic)
        plt.axis('off')
        plt.imsave('labeled.jpg', pic)

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
    def save_gcode(self):
        with open('gcode.txt','w') as f:
            for i in range(len(self.gcode)):
                f.write('%s\n'%self.gcode[i])
    #------------------------------------------------------------------------
    

if __name__=='__main__':
    input_path = 'flower.jpg'
    pic=Piture(input_path) # 輸入圖片的路徑
    pic.gray_scale()
    pic.saveImg(f'{output_name}_gray_scale')

    # pic.prewiit()
    # pic.saveImg(f'{output_name}_prewitt')
    # pic.denoise()
    # pic.sharpen()
    # pic.saveImg(f'{output_name}_sharpen')
    # pic.smooth()
    # pic.saveImg(f'{output_name}_smooth')
    # pic.edge_thinning()
    # pic.saveImg(f'{output_name}_edge_thinning')
    # pic.connect()
    # pic.saveImg(f'{output_name}_connect')
    # pic.pruning()
    # pic.saveImg(f'{output_name}_pruning')

    output_name = input_path.split('.')[0]
    pic.saveImg(f'{output_name}_gcode')
    gcode=pic.gen_gcode()
    pic.save_gcode()
  