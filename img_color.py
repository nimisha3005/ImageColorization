# -*- coding: utf-8 -*-

import numpy as np
import skimage.io as skio
from skimage import filters as f
import cv2
import os
from matplotlib import pyplot as plt
from PIL import Image
from math import *
import argparse


def load_images_from_folder(folder):
    """
    Function loads images from the folder and creates a list of images
    """

    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


def pre_processing(im, threshold=0.02):
    """
    Preprocesses the rgb plates of an image to grayscale images
    """
    skio.imshow(im)
    plt.title("base")
    plt.show()
    
    h,w,_ = im.shape
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    im1 = im.copy()
    
    # Crop grayscale image to binary form
    im1[im1<102] = 0
    im1[im1>102] = 255
    
    
    # Crop left border
    invert = cv2.bitwise_not(im1)
    cut = w
    for i in range(230):
        white_density = sum(invert[:, i])/h
        if white_density >= threshold:
            cut = i
    
    im = im[:, cut:w]
    
    
    # Crop right border
    [h, w] = im.shape
    cut = w
    for i in range(w,(w-230),-1):
        white_density = sum(invert[:, i])/h;
        if white_density >= threshold:
            cut = i
            
    im = im[:, 1:cut]
    
    
    height = np.floor(im.shape[0] / 3.0).astype(np.int)
    
    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]
    
    
    # Crop top border
    [h, w] = b.shape
    cut = h
    for i in range(230):
        white_density = sum(b[i, :])/w
        if white_density >= threshold:
            cut = i
    
    b = b[cut:h, :]
    g = g[cut:h, :]
    r = r[cut:h, :]
    
    
    # Crop bottom border
    [h, w] = r.shape
    cut = h
    for i in range(h-1,(h-230),-1):
        white_density = sum(r[i, :])/w
        if white_density >= threshold:
            cut = i
    
    b = b[1:cut, :]
    g = g[1:cut, :]
    r = r[1:cut, :]
    
    skio.imshow(b)
    plt.title("b")
    plt.show()
    
    skio.imshow(g)
    plt.title("g")
    plt.show()
    
    skio.imshow(r)
    plt.title("r")
    plt.show()
    
        
    """
    b = b[int(0.1*b.shape[1]):-int(0.1*b.shape[1]),int(0.1*b.shape[0]):-int(0.1*b.shape[0])]
    g = g[int(0.1*g.shape[1]):-int(0.1*g.shape[1]),int(0.1*g.shape[0]):-int(0.1*g.shape[0])]
    r = r[int(0.1*r.shape[1]):-int(0.1*r.shape[1]),int(0.1*r.shape[0]):-int(0.1*r.shape[0])]
    
    im1_gray = cv2.cvtColor(b[20:-20,:],cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(g[20:-20,:],cv2.COLOR_BGR2GRAY)
    im3_gray = cv2.cvtColor(r[20:-20,:],cv2.COLOR_BGR2GRAY)
    
    im1_gray = cv2.Canny(np.uint8(im1_gray),100,200)
    im1_gray = cv2.Laplacian(np.uint8(im1_gray),cv2.CV_8U)
    im2_gray = cv2.Canny(np.uint8(im2_gray),100,200)
    im2_gray = cv2.Laplacian(np.uint8(im2_gray),cv2.CV_8U)
    im3_gray = cv2.Canny(np.uint8(im3_gray),100,200)
    im3_gray = cv2.Laplacian(np.uint8(im3_gray),cv2.CV_8U)
    """
    
    im1_gray = b#cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)
    im2_gray = g#cv2.cvtColor(g,cv2.COLOR_BGR2GRAY)
    im3_gray = r#cv2.cvtColor(r,cv2.COLOR_BGR2GRAY)
    
    return im1_gray,im2_gray,im3_gray


def blue_base(im1_gray,im2_gray,im3_gray):
    """
    Aligning green and red channels considering blue channel as base
    """
    
    warp_mode = cv2.MOTION_TRANSLATION
    sz=im1_gray.shape
    
    iterations = 5000;
    epochs = 1e-10;
    #Criteria for termination of iterations
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,  epochs)
    
    #Creating warped matrix based on mode of warping being selected
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
       
    #Applying EEC transformation on green channel image and update the warped matrix
    (c, warp_matrix1) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
    
    #Applying warping technique on green channel image
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        im2_aligned = cv2.warpPerspective (im2_gray, warp_matrix1, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        im2_aligned = cv2.warpAffine(im2_gray, warp_matrix1, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    
    #Applying EEC transformation on red channel image and update the warped matrix
    (c, warp_matrix2) = cv2.findTransformECC (im1_gray,im3_gray,warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
    
    #Applying warping technique on red channel image
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        im3_aligned = cv2.warpPerspective (im3_gray, warp_matrix2, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        im3_aligned = cv2.warpAffine(im3_gray, warp_matrix2, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
   
    im_out = np.dstack([im3_aligned[40:-40,10:-10], im2_aligned[40:-40,10:-10], im1_gray[40:-40,10:-10]])
    return im_out


def green_base(im1_gray,im2_gray,im3_gray):
    """
    Aligning green and red channels considering green channel as base
    """
    
    warp_mode = cv2.MOTION_TRANSLATION
    sz=im2_gray.shape
    
    iterations = 5000;
    epochs = 1e-10;
    #Criteria for termination of iterations
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,  epochs)
    
    #Creating warped matrix based on mode of warping being selected
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    #Applying EEC transformation on blue channel image and update the warped matrix
    (c, warp_matrix1) = cv2.findTransformECC (im2_gray,im1_gray,warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
    
    #Applying warping technique on blue channel image
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        im1_aligned = cv2.warpPerspective (im1_gray, warp_matrix1, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        im1_aligned = cv2.warpAffine(im1_gray, warp_matrix1, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    
    #Applying EEC transformation on red channel image and update the warped matrix
    (c, warp_matrix2) = cv2.findTransformECC (im2_gray,im3_gray,warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
    
    #Applying warping technique on red channel image
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        im3_aligned = cv2.warpPerspective (im3_gray, warp_matrix2, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        im3_aligned = cv2.warpAffine(im3_gray, warp_matrix2, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    im_out = np.dstack([im3_aligned[40:-40,10:-10], im2_gray[40:-40,10:-10], im1_aligned[40:-40,10:-10]])
    return im_out


def red_base(im1_gray,im2_gray,im3_gray):
    """
    Aligning green and red channels considering red channel as base
    """
    
    warp_mode = cv2.MOTION_TRANSLATION
    sz=im3_gray.shape
    
    iterations = 5000;
    epochs = 1e-10;
    #Criteria for termination of iterations
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations,  epochs)
    
    #Creating warped matrix based on mode of warping being selected
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else :
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    
    #Applying EEC transformation on blue channel image and update the warped matrix
    (c, warp_matrix1) = cv2.findTransformECC (im3_gray,im1_gray,warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
    
    #Applying warping technique on blue channel image
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        im1_aligned = cv2.warpPerspective (im1_gray, warp_matrix1, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        im1_aligned = cv2.warpAffine(im1_gray, warp_matrix1, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    
    #Applying EEC transformation on green channel image and update the warped matrix
    (c, warp_matrix2) = cv2.findTransformECC (im3_gray,im2_gray,warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)
    
    #Applying warping technique on green channel image
    if warp_mode == cv2.MOTION_HOMOGRAPHY :
        im2_aligned = cv2.warpPerspective (im2_gray, warp_matrix2, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else :
        im2_aligned = cv2.warpAffine(im2_gray, warp_matrix2, (sz[1],sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    
    im_out = np.dstack([im3_gray[40:-40,10:-10], im2_aligned[40:-40,10:-10], im1_aligned[40:-40,10:-10]])
    return im_out


def white_balance(img):
    """
    Function to perform white balancing on the input image
    """

    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


def auto_contrast(img):
    """
    Function to perform auto contrast of the input image
    """
    result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(result)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result

def smooth_image(image):
    """
    Function to perform smoothing on the input image using bilateral filter
    """
    result = cv2.bilateralFilter(image,19,75,75)
    return result


def pipeline_one(image,i,name):
    """
    First pipeline to generate final image
    Original image->White balancing->Auto contrast->Smoothing->White balancing
    """
    
    cv2.imwrite(os.path.join(save_path , name, r'order_one_'+str(i)+'.jpg'), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
  
    wb_img = white_balance(image)
    cv2.imwrite(os.path.join(save_path , name, r'order_one_wb_'+str(i)+'.jpg'), cv2.cvtColor(wb_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(wb_img,(x_offset,0))
    #x_offset += image.size[0]  
    
    contrast_img = auto_contrast(wb_img)
    cv2.imwrite(os.path.join(save_path , name, r'order_one_ac_'+str(i)+'.jpg'), cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(contrast_img,(x_offset,0))
    #x_offset += image.size[0]  
    
    smooth_img = smooth_image(contrast_img)
    cv2.imwrite(os.path.join(save_path , name, r'order_one_smooth_'+str(i)+'.jpg'), cv2.cvtColor(smooth_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(smooth_img,(x_offset,0))
    #x_offset += image.size[0]  
    
    wb_img2 = white_balance(smooth_img)
    cv2.imwrite(os.path.join(save_path , name, r'order_one_wb2_'+str(i)+'.jpg'), cv2.cvtColor(wb_img2, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(wb_img2,(x_offset,0))
    #x_offset += image.size[0]  
    
    #combined.save(save_path+'combined_one'+str(i)+'.jpg')
    #combined = cv2.hconcat([image,wb_img,contrast_img,smooth_img,wb_img2])
    
    #return combined


def pipeline_two(image,i,name):
    """
    Second pipeline to generate final image
    Original image->White balancing->Auto contrast->White balancing->Smoothing
    """
    
    #combined = Image.new('RGB', (width, height))
    #x_offset = 0
    cv2.imwrite(os.path.join(save_path , name, r'order_two_'+str(i)+'.jpg'), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
  
    wb_img = white_balance(image)
    cv2.imwrite(os.path.join(save_path , name, r'order_two_wb_'+str(i)+'.jpg'), cv2.cvtColor(wb_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(wb_img,(x_offset,0))
    #x_offset += image.size[0]  
    
    contrast_img = auto_contrast(wb_img)
    cv2.imwrite(os.path.join(save_path , name, r'order_two_ac_'+str(i)+'.jpg'), cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(contrast_img,(x_offset,0))
    #x_offset += image.size[0]  
    
    wb_img2 = white_balance(contrast_img)
    cv2.imwrite(os.path.join(save_path , name, r'order_two_wb2_'+str(i)+'.jpg'), cv2.cvtColor(wb_img2, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(wb_img2,(x_offset,0))
    #x_offset += image.size[0]  
    
    smooth_img = smooth_image(wb_img2)
    cv2.imwrite(os.path.join(save_path , name, r'order_two_smooth_'+str(i)+'.jpg'), cv2.cvtColor(smooth_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(smooth_img,(x_offset,0))
    #x_offset += image.size[0]  
    
    #combined.save(save_path+'combined_two'+str(i)+'.jpg')
    #combined = cv2.hconcat([image,wb_img,contrast_img,wb_img2,smooth_img])
        
    #return combined

    
def pipeline_three(image,i,name):
    """
    Third pipeline to generate final image
    Original image->Auto contrast->White balancing->Auto contrast->Smoothing
    """
    #combined = Image.new('RGB', (width, height))
    #x_offset = 0
    cv2.imwrite(os.path.join(save_path , name, r'order_three_'+str(i)+'.jpg'), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
  
    contrast_img = auto_contrast(image)
    cv2.imwrite(os.path.join(save_path , name, r'order_three_ac_'+str(i)+'.jpg'), cv2.cvtColor(contrast_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(contrast_img,(x_offset,0))
    #x_offset += image.size[0]  
    
    wb_img = white_balance(contrast_img)
    cv2.imwrite(os.path.join(save_path , name, r'order_three_wb_'+str(i)+'.jpg'), cv2.cvtColor(wb_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(wb_img,(x_offset,0))
    #x_offset += image.size[0]  
    
    contrast_img2 = auto_contrast(wb_img)
    cv2.imwrite(os.path.join(save_path , name, r'order_three_ac2_'+str(i)+'.jpg'), cv2.cvtColor(contrast_img2, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(contrast_img,(x_offset,0))
    #x_offset += image.size[0]  
    
    smooth_img = smooth_image(contrast_img2)
    cv2.imwrite(os.path.join(save_path , name, r'order_three_smooth_'+str(i)+'.jpg'), cv2.cvtColor(smooth_img, cv2.COLOR_BGR2RGB))
    cv2.waitKey(0)
    #combined.paste(smooth_img,(x_offset,0))
    #x_offset += image.size[0]  
    
    #combined.save(save_path+'combined_three'+str(i)+'.jpg')
    #combined = cv2.hconcat([image,contrast_img,wb_img,contrast_img2,smooth_img])
    
    #return combined
import numpy as np
from statistics import mean
def circshift(im,y,x):
    a=np.roll(im,y)
    b=np.roll(a,x)
    return b

def findshift(im1,im2,s):
    best = float('inf')
    for dy in range(-s,s):
        for dx in range(-s,s):
            shifted = circshift(im2, dy, dx)      
            score = sum(sum((im1-shifted)**2))
            #print(mean(score),score.shape)
            if (score) <= best:
                best = (score)
                shift = [dy, dx]
    return shift

def scale_img(img,factor):
    blur = f.gaussian(img, sigma=(3, 3), truncate=factor, multichannel=True)
    res = cv2.resize(blur, (int(blur.shape[1]*(1/factor)),int(blur.shape[0]*(1/factor))), interpolation = cv2.INTER_AREA)
    return res
             
        
def read_images(images,names):
    """
    This function is responsible for main processing of images    
    """
    
    count=1
    for i in images:
        im = i
        name = names[count-1]
        blue,green,red = pre_processing(im)
        
        b_shift = findshift((green), (blue), 2)
        r_shift = findshift((green), (red), 2)
        print("shift",b_shift,r_shift)

        im_b = circshift(blue, b_shift[0], b_shift[1])
        im_r = circshift(red, r_shift[0],r_shift[1])
        
        im_b = np.uint8(im_b);
        im_r = np.uint8(im_r);

        skio.imshow(red)
        plt.title("base")
        plt.show()
        
        skio.imshow(blue)
        plt.title("base")
        plt.show()
        
        skio.imshow(im_r)
        plt.title("base")
        plt.show()
        
        skio.imshow(im_b)
        plt.title("base")
        plt.show()
        

        im = np.dstack([im_r, green, im_b])
        
        cv2.imwrite(os.path.join(save_path , name, r'aligned.jpg'),cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        skio.imshow(im)
        plt.title("base")
        plt.show()
        
        
        
        blue_img = blue_base(blue,green,red)
        green_img = green_base(blue,green,red)
        red_img = red_base(blue,green,red)
    
        
        skio.imshow(blue_img)
        plt.title("Blue base")
        plt.show()
        
        skio.imshow(green_img)
        plt.title("Green base")
        plt.show()
        
        skio.imshow(red_img)
        plt.title("Red base")
        plt.show()
        
        pipeline_one(green_img,count,name)     
        pipeline_two(green_img,count,name)     
        pipeline_three(green_img,count,name)     
        
        """
        skio.imshow(image_one)
        plt.title("pipeline 1")
        plt.show()
        
        skio.imshow(image_two)
        plt.title("pipeline 2")
        plt.show()
        
        skio.imshow(image_three)
        plt.title("pipeline 3")
        plt.show()
        """
        
        print(cv2.imwrite(os.path.join(save_path , name, r'blue_base'+str(count)+'.jpg'),cv2.cvtColor(blue_img, cv2.COLOR_BGR2RGB)))
        cv2.waitKey(0)
        print(cv2.imwrite(os.path.join(save_path , name, r'green_base'+str(count)+'.jpg'),cv2.cvtColor(green_img, cv2.COLOR_BGR2RGB)))
        cv2.waitKey(0)
        print(cv2.imwrite(os.path.join(save_path , name, r'red_base'+str(count)+'.jpg'),cv2.cvtColor(red_img, cv2.COLOR_BGR2RGB)))
        cv2.waitKey(0)
        
        count=count+1


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--Input", help = "Path to input images")
    parser.add_argument("-o", "--Output", help = "Path to save results")
    
    args = parser.parse_args()
    
    S_PATH =  args.Output #path where need to save generated images
    I_PATH = args.Input #path where input images present
    save_path = S_PATH
    input_path = I_PATH
    
    #Read images and create their list
    imgs = load_images_from_folder(input_path)
    
    #Folder to save final results
    folder_names=['one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']
    
    read_images(imgs,folder_names)
    
