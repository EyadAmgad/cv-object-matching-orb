import cv2
import numpy as np
import matplotlib.pyplot as plt


query_img = cv2.imread('images/QueryFoot.png', cv2.IMREAD_GRAYSCALE)   # object to be found
target_img = cv2.imread('images/TargetFoot.png', cv2.IMREAD_GRAYSCALE) # image where object exists


orb = cv2.ORB_create(nfeatures=1000)


kp1, des1 = orb.detectAndCompute(query_img, None)
kp2, des2 = orb.detectAndCompute(target_img, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)


matched_img = cv2.drawMatches(query_img, kp1, target_img, kp2, matches[:50], None, flags=2)


plt.figure(figsize=(15, 10))
plt.imshow(matched_img)
plt.title('Object Matching using ORB')
plt.axis('off')
plt.savefig('output/matched_keypoints.png')  
plt.show()  

plt.show()
