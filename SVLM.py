#   Trying a Python code for paper
#   A Space-Variant Luminance Map based Color Image Enhancement
#   https://ieeexplore.ieee.org/abstract/document/5681151


import numpy as np
import cv2

image = cv2.imread("test9.jpg")


assert image.shape[2] == 3

# Split channels and converting source image for gray scale

B = np.double(image[:,:,0])
G = np.double(image[:,:,1])
R = np.double(image[:,:,2])
Gray = 0.299*R + 0.587*G + 0.114*B



height,width= Gray.shape
print(Gray.shape)


# Applying a Gaussian  mask  to  extract  the  local  characteristic 
# For 1/2, 1/4, 1/8 and After resizing images for original size
sigmablur = 20.5
sizegaussiankernel = 3
L1 = cv2.GaussianBlur(Gray,(sizegaussiankernel,sizegaussiankernel),sigmablur)
L2 = cv2.resize(cv2.GaussianBlur(cv2.resize(Gray,(int(height/2),int(width/2))),(sizegaussiankernel,sizegaussiankernel),sigmablur), (width,height),interpolation=cv2.INTER_CUBIC)
L3 = cv2.resize(cv2.GaussianBlur(cv2.resize(Gray,(int(width/4),int(height/4))),(sizegaussiankernel,sizegaussiankernel),sigmablur), (width,height),interpolation=cv2.INTER_CUBIC)
L4 = cv2.resize(cv2.GaussianBlur(cv2.resize(Gray,(int(width/8),int(height/8))),(sizegaussiankernel,sizegaussiankernel),sigmablur), (width,height),interpolation=cv2.INTER_CUBIC)
SVLM = (L1 + L2 + L3 + L4)/4




# Luminance Enhancement
# Gamma correction 

alfa = 1 #  Global image dependency(0-1)

gamma = np.double(alfa**((128 - np.double(SVLM))/128))
O = 255*((np.double(Gray)/255)**gamma)



# Contrast Enhancement 
sigma = np.std(Gray, ddof=1)
if sigma <= 40:
    P = 2
elif sigma > 40 and sigma <= 80:
    P = ((-0.025*sigma) + 3) 
else:
    P=1


# Color Restoration  
E = (np.double(SVLM)/O)**np.double(P)
S = 255*((O/255)**E)

# Adjust factor for the color hue
lambda_red = 0.9
lambda_green = 0.9
lambda_blue = 0.9
Rm = S**(R/O)*lambda_red
Gm = S**(G/O)*lambda_green
Bm = S**(B/O)*lambda_blue



enhanced_image = np.zeros((image.shape))


enhanced_image[:,:,0] = Rm
enhanced_image[:,:,1] = Gm
enhanced_image[:,:,2] = Bm
enhanced_image = ((enhanced_image/255)*1.25)




while(1):
    cv2.imshow("Original Image",image)
    cv2.imshow("Image",enhanced_image)
    k = cv2.waitKey(33)
    if k==27:   
        break
        cv2.destroyAllWindows()
    elif k==-1:  
        continue
