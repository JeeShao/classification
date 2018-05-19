import cv2,shutil

src = "./trainImgs/lighter/1.jpg"
to = "./trainImgs/lighter/"

img = cv2.imread(src,cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img,(64,64))
cv2.imwrite(src,img)
for i in range(2,301):
    shutil.copyfile(src,to+"%d.jpg" %(i))