
import cv2
import pytesseract as pt
import numpy as np

class Plate_recognition:
    def __init__(self, config_path, img_path) -> None:
        self.harcascade = cv2.CascadeClassifier(config_path)
        self.img = cv2.imread(img_path)
        self.min_area = 500
        self.img_roi = None

    def extract_text(self):
        gray = cv2.cvtColor(self.img_roi, cv2.COLOR_BGR2GRAY)
        img = np.array(gray)
        _, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)

        no_plate = pt.image_to_string(img)
        print(no_plate)

    def scan_image(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        plates = self.harcascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=3, minSize=(30,30))
        for (x,y,w,h) in plates:
            area = w*h
            if area > self.min_area:
                cv2.rectangle(self.img, (x,y), (x+w, y+h), (0,255,0),2)
                cv2.putText(self.img, "number plates",(x,y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)

                self.img_roi = self.img[y:y+h, x:x+w]
        cv2.imshow('img', self.img)
        key = cv2.waitKey(0)
        if key & 0xFF == ord('p'):
            self.extract_text()
        if key ==27:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    harcascade = "./model/haarcascade_russian_plate_number.xml"
    img = './imgs/car.jpeg'
    plate_reco = Plate_recognition(harcascade, img)
    plate_reco.scan_image()
