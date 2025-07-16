import os
import cv2


def convert_weak(imgpath, savepath):
    image = cv2.imread(imgpath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)
    cv2.imwrite(savepath, image)


path = r"E:\dataset\CLCD\test\label"
save = r"E:\dataset\CLCD\test\label_weak"
i = 1
for item in os.listdir(path):
    print(f'{i}/{len(os.listdir(path))}, {item}')
    img = path + '/' + item
    save_path = save + '/' + item
    convert_weak(img, save_path)
    i = i + 1


