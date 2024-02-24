import numpy as np
from PIL import Image
from  pylab import *
import matplotlib.pyplot as  plt
import time as tm
from skimage.filters.rank import entropy
from skimage.morphology import disk

#DNA BASE ENCRIPTION FUNCTION
def DNA_ENC(x1,x2,ch):
    if ch==0:
        if(x1==0 and x2==0):
            p='A'
         
        if(x1==1 and x2==0):
            p='C'
        if(x1==0 and x2==1):
            p='G'
        if(x1==1 and x2==1):
            p='T' 
    if ch==1:
        if(x1==0 and x2==0):
            p='A'
         
        if(x1==0 and x2==1):
            p='C'
          
        if(x1==1 and x2==0):
            p='G'
        if(x1==1 and x2==1):
            p='T'
    if ch==2:
        if(x1==0 and x2==1):
            p='A'
         
        if(x1==1 and x2==1):
            p='C'
          
        if(x1==0 and x2==0):
            p='G'
        if(x1==1 and x2==0):
            p='T'
    if ch==3:
        if(x1==0 and x2==1):
            p='A'
         
        if(x1==0 and x2==0):
            p='C'
          
        if(x1==1 and x2==1):
            p='G'
        if(x1==1 and x2==0):
            p='T'
    if ch==4:
        if(x1==1 and x2==0):
            p='A'
         
        if(x1==1 and x2==1):
            p='C'
          
        if(x1==0 and x2==0):
            p='G'
        if(x1==0 and x2==1):
            p='T'
    if ch==5:
        if(x1==1 and x2==0):
            p='A'
         
        if(x1==0 and x2==0):
            p='C'
          
        if(x1==1 and x2==1):
            p='G'
        if(x1==0 and x2==1):
            p='T'
    if ch==6:
        if(x1==1 and x2==1):
            p='A'
         
        if(x1==1 and x2==0):
            p='C'
          
        if(x1==0 and x2==1):
            p='G'
        if(x1==0 and x2==0):
            p='T'
    if ch==7:
        if(x1==1 and x2==1):
            p='A'
         
        if(x1==0 and x2==1):
            p='C'
          
        if(x1==1 and x2==0):
            p='G'
        if(x1==0 and x2==0):
            p='T'
    return(p)

#DNA BASE DECRIPTION FUNCTION
def DNA_DEC(x1,ch):
    if ch==0:
        s=""
        if(x1=='A'):
            s=s+'00'
        if(x1=='C'):
            s=s+'10'
        if(x1=='G'):
            s=s+'01'
        if(x1=='T'):
            s=s+'11'
    if ch==1:
        s=""
        if(x1=='A'):
            s=s+'00'
        if(x1=='C'):
            s=s+'01'
        if(x1=='G'):
            s=s+'10'
        if(x1=='T'):
            s=s+'11'
    if ch==2:
        s=""
        if(x1=='A'):
            s=s+'01'
        if(x1=='C'):
            s=s+'11'
        if(x1=='G'):
            s=s+'00'
        if(x1=='T'):
            s=s+'10'
    if ch==3:
        s=""
        if(x1=='A'):
            s=s+'01'
        if(x1=='C'):
            s=s+'00'
        if(x1=='G'):
            s=s+'11'
        if(x1=='T'):
            s=s+'10'
    if ch==4:
        s=""
        if(x1=='A'):
            s=s+'10'
        if(x1=='C'):
            s=s+'11'
        if(x1=='G'):
            s=s+'00'
        if(x1=='T'):
            s=s+'01'
    if ch==5:
        s=""
        if(x1=='A'):
            s=s+'10'
        if(x1=='C'):
            s=s+'00'
        if(x1=='G'):
            s=s+'11'
        if(x1=='T'):
            s=s+'01'
    if ch==6:
        s=""
        if(x1=='A'):
           s=s+'11'
        if(x1=='C'):
            s=s+'10'
        if(x1=='G'):
            s=s+'01'
        if(x1=='T'):
            s=s+'00'
    if ch==7:
        s=""
        if(x1=='A'):
           s=s+'11'
        if(x1=='C'):
            s=s+'01'
        if(x1=='G'):
            s=s+'10'
        if(x1=='T'):
            s=s+'00'
    return(s)

#CONVERTION FROM  DECIMAL NUMBER TO BINARY NUMBER
def DEC_BIN(n):
   binary = format(n, "08b")
   return(binary)


#CONVERTION FROM  BINARY NUMBER TO DECIMAL NUMBER 
def BIN_DEC(binary):
    decimal = 0
    for digit in binary:
        decimal = decimal*2 + int(digit)
    return(decimal)
#CONVERTION FROM  BINARY NUMBER TO DECIMAL NUMBER 
def BIN_DEC_REV(binary):
    decimal = 0
    for digit in range(7,-1,-1):
        decimal = decimal*2 + int(binary[digit])
    return(decimal)


#read color image
ii=Image.open(r"C:\Users\Rapunzel\Desktop\project\xray.jpg")
#resize color image
ii = ii.resize((64,64))
#convert color image to gray image
im=array(ii.convert('L'))
m,n=im.shape

t=tm.time()

A=im.flatten()
x1=0.449
lmd=3.989
B=zeros(m*n)
B1=zeros(m*n)
B2=zeros(m*n)
A1=zeros(m*n)
A2=zeros(m*n)
A3=zeros(m*n)
im1=np.zeros((m,n))
im2=np.zeros((m,n))
im3=np.zeros((m,n))
for i in range(0,m*n):
    x2=lmd*x1*(1-x1)
    B[i]=x2
    x1=x2
print("B",B)
B1=np.sort(B)
print("B1",B1)
for i in range(0,m*n):
    for j in range(0,m*n):
        if B[i]==B1[j]:
            B2[i]=j

print("B2",B2)

for i in range(0,m*n):
    k=round(B2[i])
    A1[i]=A[k]
#print("A",A)
ss = np. reshape(A, (m, n))
print("original image")
print(ss)
#print(A1)
s = np. reshape(A1, (m, n))
print("chaotic encryption ")
print(s)

dna=""
for i in range(0,m):
    for j in range(0,n):
        x=int(s[i,j]);
        ch=x%8
        #print(x)
        y=DEC_BIN(x)
        #print(y)
        yy=BIN_DEC_REV(y)       
        #print(yy)
        im1[i,j]=yy
        xx=DEC_BIN(yy)
        #print(xx)
        x2=""
        for k in range(1,8,2):
            x2=x2+DNA_ENC(int(xx[k-1]),int(xx[k]),ch)
        #print(x2)
        dna=dna+x2
        t1=tm.time()
        sss=""
        for k in range(0,4):
            sss=sss+DNA_DEC(x2[k],ch)
        #print(s)
        z=BIN_DEC(sss)
        im2[i,j]=z
        #print(z)
        zz=DEC_BIN(z)       
        #print(zz)
        zzz=BIN_DEC_REV(zz)
        #print(zzz)
        im3[i,j]=zzz

print("bit shift Image")
print(im1)
print(" DNA Sequence and Length of DNA")
print(dna,len(dna))
print(" bit reverse shift Image")
print(im2)
print("chaotic Decrypted Image")
print(im3)
 
k=0
for i in range(0,m):
    for j in range(0,n):
        A2[k]=im3[i,j]
        k=k+1
#print(A2)


#decryption

for i in range(0,k):
    for j in range(0,k):
        if i==B2[j]:
            A3[i]=A2[j]
#print(A3)
       
s1 = np. reshape(A3, (m, n))
print("Final Decrypted Image")
print(s1)
t1=tm.time()

print("Encryption and Decryption time",(t1-t))


plt.subplot(3,3,1)
plt.imshow(ii)
plt.title("Original")
#plt.show()
plt.subplot(3,3,2)
SS=Image.fromarray(ss)
plt.imshow(SS.convert('RGB'))
plt.title("Gray image  ")
#plt.show()

plt.subplot(3,3,3)
S=Image.fromarray(s)
plt.imshow(S.convert('RGB'))
plt.title("Chaostic encrypt image  ")
#plt.show()

plt.subplot(3,3,4)
IM1=Image.fromarray(im1)
plt.imshow(IM1.convert('RGB'))
plt.title("Bit shift Image")
#plt.show()

plt.subplot(3,3,5)
im21=Image.fromarray(im2)
plt.imshow(im21.convert('RGB'))
plt.title("Bit reverse shift image")    
#plt.show()

plt.subplot(3,3,6)
im31=Image.fromarray(im3)
plt.imshow(im31.convert('RGB'))
plt.title("Chaostic decrypt image")
#plt.show()         

plt.subplot(3,3,7)
s1=Image.fromarray(s1)
plt.imshow(s1.convert('RGB'))
plt.title("Final decrypted image")
plt.show()

#plt.subplot(3,3,8)
#plt.imshow(ii)
#plt.title("Final Original Image")
#plt.show()

plt.title("Histogram of original RED and encrypted BLUE Image")
plt.xlabel("gray level values")
plt.ylabel("gray level count")
histogram, bin_edges = np.histogram(im, bins=256, range=(0, 256))
plt.plot( histogram,color="red")
histogram1, bin_edges1= np.histogram(im2, bins=256, range=(0, 256))
plt.plot( histogram1, color="blue")
plt.show()


r = np.sum(np.cov(SS, IM1)/(m*n))/(np.var(SS)*np.var(IM1))
print("correlation",r)

mse = np.mean((im1 - ss) ** 2)
psnr = 20 * log10(255.0 / sqrt(mse))
print("psnr",psnr)

im1=array(IM1.convert('L'))
entr_img1 =np.sum(entropy(im1, disk(10)))/(m*n)
im=array(ii.convert('L'))
entr_img2 =np.sum(entropy(im, disk(10)))/(m*n)
print(entr_img1-entr_img2)
