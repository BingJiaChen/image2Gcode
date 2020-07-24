import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import potrace

class Piture():
    def __init__(self,filepath):
        # self.img=mpimg.imread(filepath)
        self.img = Image.open(filepath)
        # self.img = self.img.resize((300,300))
        self.img = np.array(self.img)
        print(self.img.shape)
        self.h,self.w,self.c=self.img.shape
        self.pre=np.ones(self.img.shape)
        self.gcode=['G28']
        self.connectPixel = 10 # connection expand max
        self.x_max=50
        self.y_max=50
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
        return result
    #------------------------------------------------------------------------
    def resizeAfterGrayScale(self, size):
        print('Resize to: ', size)
        tmp = self.pre[:, :, 0]
        tmp = Image.fromarray(np.uint8(tmp * 255), 'L')
        tmp = tmp.resize(size)
        tmp.show()
        tmp = np.array(tmp)
        self.pre = np.zeros((tmp.shape[0], tmp.shape[1], 3))
        self.h,self.w = tmp.shape
        for i in range(tmp.shape[0]):
            for j in range(tmp.shape[1]):
                G = 0
                if tmp[i, j] > 0: G = 1
                self.pre[i, j] = np.array([G, G, G])

    def denoise(self):
        print('start to denoise...')
        for i  in range(1,self.h):
            for j in range(1,self.w):
                if  self.pre[i,j,0]==1 and np.sum(self.pre[i-1:i+2,j-1:j+2,0])==1:
                    self.pre[i,j]=np.array([0,0,0])
        return self.pre        

    def edge_thinning(self):
        print('start edge thinning...')
        deletable = np.zeros(self.pre.shape)
        thin_times = 4
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
    def pruning(self):
        print('start pruning...')
        deletable = np.zeros(self.pre.shape)
        # time = 0
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
            # time +=1
            self.pre=self.pre-deletable
            print('pruning for',times + 1,'times , deleting',np.sum(deletable)/3,"pixels")
            if np.sum(deletable)==0:
                break
            deletable = np.zeros(self.pre.shape)
            
            # if time==self.connectPixel:
            #     break
        return self.pre

    def show(self):
        plt.imshow(self.pre)
        plt.axis('off')
        plt.show()

    def saveImg(self, output):
        plt.imshow(self.pre)
        plt.axis('off')
        plt.imsave(output + '.jpg', self.pre)
        print('Save ' + output + '.jpg')

    def gen_gcode(self):
        print('generate gcode...')
        bmp=potrace.Bitmap(self.pre[:,:,0])
        path=bmp.trace()
        flag = 0
        for curve in path:
            
            ratio=self.x_max/max(self.w,self.h) #normalize for drawing machine
            self.gcode.append('M280 P0 S60') #抬筆
            self.gcode.append('G0 X%.4f Y%.4f'%(curve.start_point[0]*ratio,curve.start_point[1]*ratio)) #移動到起始點
            self.gcode.append('M280 P0 S0') #下筆
            for segment in curve:
                # print(segment)
                if segment.is_corner:
                    self.gcode.append('G1 X%.4f Y%.4f'%(segment.c[0]*ratio,segment.c[1]*ratio)) #畫至corner的轉角點
                    self.gcode.append('G1 X%.4f Y%.4f'%(segment.end_point[0]*ratio,segment.end_point[1]*ratio)) #畫至corner的終點
                else:
                    self.gcode.append('G1 X%.4f Y%.4f'%(segment.end_point[0]*ratio,segment.end_point[1]*ratio)) #畫至Bezier segment的終點
                    # if flag%4==0:
                    #     self.gcode.append('G1 X%.4f Y%.4f'%(segment.end_point[0]*ratio,segment.end_point[1]*ratio)) #畫至Bezier segment的終點
                    #     flag+=1
                    # else:
                    #     flag+=1
        self.gcode.append('M280 P0 S60') #抬筆
        return self.gcode
    
    def save_gcode(self):
        with open('output.txt','w') as f:
            for i in range(len(self.gcode)):
                f.write('%s\n'%self.gcode[i])

if __name__=='__main__':
    pic=Piture('img/bear.jpg') #輸入圖片的路徑
    pic.gray_scale()
    pic.saveImg('gray_scale')
    pic.prewiit()
    pic.saveImg('prewitt')
    pic.denoise()
    # pic.edge_thinning()
    # pic.denoise()
    # pic.saveImg('edge_thinning')
    # pic.denoise()
    pic.resizeAfterGrayScale((100, 100))
    pic.resizeAfterGrayScale((600, 600))
    pic.saveImg('resized')
    # pic.connect()
    # pic.saveImg('connect')
    # pic.pruning()
    # pic.saveImg('pruning')
    pic.saveImg('To Gcode')
    gcode=pic.gen_gcode()
    pic.save_gcode()

