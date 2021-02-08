import numpy as np
import cv2
from osgeo import gdal
import pdb
import sys
sys.path.append('../src/')
import deb

def load_image(patch):
    # Read Image
    print (patch)
    gdal_header = gdal.Open(patch)
    # get array
    img = gdal_header.ReadAsArray()
    return img
labels_path = 'labels/'
mask_path = ''
mask = cv2.imread(mask_path + 'TrainTestMask.tif', 0).astype(np.uint8)
label = load_image(labels_path + '20160731_S1.tif').astype(np.uint8)

deb.prints(mask.shape)
deb.prints(label.shape)

#label_train = label[]

# lem1 train mask
deb.prints(np.unique(mask, return_counts=True))
deb.prints(np.unique(label, return_counts=True))

mask_train = mask.copy()
mask_train[mask_train!=1] = 0

label_train = label.copy()
label_train[mask!=1] = 0
deb.prints(np.unique(label_train, return_counts=True))

pdb.set_trace()

mask_train_255 = mask_train.copy()
mask_train_255[mask_train_255>0] = 255

print("Unique mask", np.unique(mask, return_counts=True))
print("Unique lem2_label", np.unique(lem2_label, return_counts=True))
print("Unique lem2_mask", np.unique(lem2_mask, return_counts=True))

# find lem2 contours

#_,lem2_mask = cv2.threshold(lem2_mask, 1, 255,  
#                            cv2.THRESH_BINARY) 
#print(np.unique(lem2_mask, return_counts=True))

_, contours, _ =cv2.findContours(mask_train_255, cv2.RETR_TREE, 
                            cv2.CHAIN_APPROX_SIMPLE) 
print(len(contours))
#pdb.set_trace()

mask_val = np.zeros(mask.shape, np.uint8)
val_count = 0
train_count = 0


#pdb.set_trace()
for cnt in contours : 

    # create each polygon mask
    cnt_mask = np.zeros(mask.shape, np.uint8)
    #cv2.drawContours(cnt_mask, cnt, -1, 255, -1)
    cv2.fillPoly(cnt_mask, pts =[cnt], color=255)
    ##print("np.average(cnt_mask)", np.average(cnt_mask))
    # check if it touches lem1 train mask

    area = cv2.contourArea(cnt)
    ##print("area", area)
    mean_train = cv2.mean(mask_train, mask=cnt_mask)
    ##print("mean", mean)

    edge_mean = cv2.mean(edge_mask, mask=cnt_mask)
    ##print("edge_mean", edge_mean)
    mean_test = cv2.mean(mask_test, mask=cnt_mask)
    coffee_id = 4
    lem2_label_coffee = lem2_label.copy()
    lem2_label_coffee[lem2_label_coffee!=coffee_id] = 0
    mean_test_coffee = cv2.mean(lem2_label_coffee, mask=cnt_mask)
    #pdb.set_trace()
    coffee_set = -1
    if np.average(mean_test_coffee)>0.0:
        coffee_set = np.random.randint(2)
        

    if area>200 and np.average(edge_mean) == 0.0:
        if np.average(mean_train) > 0.0 or coffee_set == 0:
            cv2.fillPoly(lem2_test_mask, pts =[cnt], color=1)
            train_count = train_count + 1
#        elif np.average(mean_test) > 0.0:
        elif np.average(mean_test) > 0.0 or coffee_set == 1:
            #cv2.drawContours(lem2_test_mask, cnt, -1, 255, -1)
            cv2.fillPoly(lem2_test_mask, pts =[cnt], color=2) # 2 for testing
            test_count = test_count + 1
            ##print(cnt)
            ##cv2.imwrite('image.png',cnt_mask)
            ##pdb.set_trace()
        else:
            rand_n = np.random.randint(4)

            # if coffee class, use it.

            if rand_n == 0:
                #cv2.drawContours(lem2_test_mask, cnt, -1, 255, -1)
                cv2.fillPoly(lem2_test_mask, pts =[cnt], color=2) # 2 for testing
                test_count = test_count + 1
                ##print(cnt)
                ##cv2.imwrite('image.png',cnt_mask)
                ##pdb.set_trace()
print(train_count, test_count)
cv2.imwrite('TrainTestMask_large_coffee.tif', lem2_test_mask)
print("np.unique(lem2_test_mask, return_counts=True)", np.unique(lem2_test_mask, return_counts=True))
 
   
    # Shortlisting the regions based on there area. 
#    if area > 400:  
#        approx = cv2.approxPolyDP(cnt,  
#                                  0.009 * cv2.arcLength(cnt, True), True) 
   
#        # Checking if the no. of sides of the selected region is 7. 
#        if(len(approx) == 7):  
#            cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5) 
   
# Showing the image along with outlined arrow. 
