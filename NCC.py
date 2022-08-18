
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as pl





def Normalized_Cross_Correlation(roi, target):
   cor = np.sum(roi * target)
   nor = np.sqrt( (np.sum(roi ** 2))) * np.sqrt(np.sum(target ** 2))
   #print(cor/nor)
   return cor / nor


def tm(img, target):
  
   img_height, img_width = img.shape                                             
   tar_height, tar_width = target.shape
   (max_Y, max_X) = (0, 0)
   MaxValue = 0


   NccVal = np.zeros((img_height-tar_height, img_width-tar_width))
  
   img = np.array(img, dtype="int")
   target = np.array(target, dtype="int")

   
   for y in range(0, img_height-tar_height):
     
      
      for x in range(0, img_width-tar_width):
          
           roi = img[y : y+tar_height, x : x+tar_width]
           NccVal[y, x] = Normalized_Cross_Correlation(roi, target)
         
           if NccVal[y, x] > 0.975:
               MaxValue = NccVal[y, x]
               (max_Y, max_X) = (y, x)
               print(MaxValue)
               top_left = (max_X, max_Y)
               cv2.rectangle(image, top_left, (top_left[0] + target.shape[1], top_left[1] + target.shape[0]), 0, 3)
   return (max_X, max_Y)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to input image")
    ap.add_argument("-t", "--target", required = True, help = "Path to target")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"], 0)
    
    target = cv2.imread(args["target"], 0)


    #cv2.imshow("image", image)
    #cv2.waitKey(0)
    height, width = target.shape
    top_left = tm(image, target)
    #draws a rectangle based on the info received 
    target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(image, top_left, (top_left[0] + width, top_left[1] + height), 0, 9)
    pl.subplot(111)
    pl.imshow(image)
    pl.title('result')
    pl.show()
