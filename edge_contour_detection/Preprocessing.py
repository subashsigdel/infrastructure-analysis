import cv2
import numpy as np
import matplotlib.pyplot as plt

class Preprocessing:
    def __init__(self, image_path):
        self.image_path = image_path

        # internal storage
        self.img_color = None
        self.img_rgb = None
        self.img_gray = None

    def run(self):

        self.load_image()
        self.preprocess()
        self.detect_regions()
        self.detect_boundaries()
        self.remove_internal_edges()
        self.extract_contours()
        self.final_output()
        self.statistics()


    def load_image(self):
        self.img_color = cv2.imread(self.image_path)
        if self.img_color is None:
            raise FileNotFoundError("Cannot load image")

        self.img_rgb   = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2RGB)
        self.img_gray  = cv2.cvtColor(self.img_color, cv2.COLOR_BGR2GRAY)


    def preprocess(self):
        
        clahe = cv2.createCLAHE(2.0, (8,8))
        enhanced = clahe.apply(self.img_gray)
        self.blur = cv2.GaussianBlur(enhanced, (15,15), 0)

    def detect_regions(self):

        thresh = cv2.adaptiveThreshold(
            self.blur,255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            181,2
        )

        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        r = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel_close,iterations=3)

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        r = cv2.morphologyEx(r,cv2.MORPH_OPEN,kernel_open,iterations=2)

        contours,_ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.large_regions = [c for c in contours if cv2.contourArea(c)>3000]

        self.regions = np.zeros_like(self.img_gray)
        cv2.drawContours(self.regions, self.large_regions, -1, 255, -1)

    def detect_boundaries(self):

        e = cv2.Canny(self.regions,50,150)

        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT,(11,1))
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT,(1,11))

        h = cv2.morphologyEx(e,cv2.MORPH_CLOSE,kernel_h)
        v = cv2.morphologyEx(e,cv2.MORPH_CLOSE,kernel_v)
        c = cv2.bitwise_or(h,v)

        close = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
        c = cv2.morphologyEx(c,cv2.MORPH_CLOSE,close,iterations=2)

        dil = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        self.edges_connected = cv2.dilate(c,dil,1)

    def remove_internal_edges(self):

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        self.field_interiors = cv2.erode(self.regions,kernel,1)

        self.edges_external_only = self.edges_connected.copy()
        self.edges_external_only[self.field_interiors>0]=0

    def extract_contours(self):

        contours,_ = cv2.findContours(self.edges_external_only,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

        self.filtered_contours=[]

        for c in contours:

            per = cv2.arcLength(c,True)
            area = cv2.contourArea(c)

            if per<80: continue
            comp = 4*np.pi*area/(per*per)
            if comp>0.5: continue

            eps=0.003*per
            self.filtered_contours.append(cv2.approxPolyDP(c,eps,True))

    def final_output(self):

        mask = np.zeros_like(self.edges_connected)
        cv2.drawContours(mask,self.filtered_contours,-1,255,2)

        img = self.img_rgb.copy()
        img[mask>0]=[255,0,0]

        plt.figure(figsize=(12,8))
        plt.imshow(img)
        plt.title("Detected Field Boundaries")
        plt.axis("off")
        plt.show()

    def statistics(self):
        print()
        print("FIELD BOUNDARY DETECTION RESULTS")
        print(f"Detected regions : {len(self.large_regions)}")
        print(f"Final boundaries : {len(self.filtered_contours)}")


if "__main__"==__name__:
    detector = Preprocessing("images/tile_82944_46080.png")
    detector.run()