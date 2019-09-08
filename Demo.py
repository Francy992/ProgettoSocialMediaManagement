from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt
import os, pathlib, sys
import cv2
import time

from AlexNet import *
from ImportFile import *


#img_test = "./Dataset/Test/Scafiti (11).jpeg"
#name, cl = an.get_prediction(img_test, 3)
#print("Persona presente in foto: ", name, ", classe predetta: ", cl)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class Demo(QMainWindow, QWidget):
    def __init__(self):
        super().__init__()
        self.net = AlexNet(train = False, dataset_path="", pth="Train_alexnet_lr0.0001_momentum0.8_epochs53.pth")
        self.net.load_model("Train_alexnet_lr0.0001_momentum0.8_epochs53")
        #self.net.fit_knn()

        self.setWindowTitle('Demo')
        self.lbl = QLabel(self)
        self.rsize_x = 256 * 2
        self.rsize_y = 256 * 2
        self.canvas = None

        self.setGeometry(100, 100, 800, 600)
        #self.setWindowIcon(QIcon(resource_path(path_to_image + 'icon.png')))
        #self.setStyleSheet("background-color: #423f3f; QText{color: #b4acac}")

        self.import_img_btn()
        #self.import_video_btn().move(0,30)

        self.show()

    def show_image(self, img):
        rgba_img = img.convert("RGBA")
        qim = ImageQt(rgba_img)
        pix = QPixmap.fromImage(qim)
        self.lbl.deleteLater()
        self.lbl = QLabel(self)
        self.lbl.setPixmap(pix)
        self.lbl.resize(pix.width(), pix.height())
        width = self.geometry().width()
        height = self.geometry().height()
        self.lbl.move(width / 2 - pix.width() / 2, height / 2 - pix.height() / 2)
        self.lbl.updateGeometry()
        self.lbl.update()
        self.update()
        self.lbl.show()

    def import_img_btn(self):
        importAct = self.button(self.process_image,"import image")
        return importAct

    def import_video_btn(self):
        importAct = self.button(self.process_video,"import video")
        return importAct

    def button(self, function, text):
        btn = QPushButton(text, self)
        btn.clicked.connect(function)
        return btn

    def import_file(self,):
        imp = ImportFile()

        self.path = imp.openFileNameDialog()
        if self.path == "wrong path": print("Nessun file caricato")
        print("path: ", self.path)

    '''def process_video(self):
        self.import_file()

        if self.path != "wrong path":
            try:
                video = cv2.VideoCapture(self.path)
                success = True

                while(success):
                    success, img = video.read()

                    if success:
                        img_pil = Image.fromarray(img)
                        img_pil = img_pil.resize((self.net.width,self.net.height))
						print("net classification1")
						name, cl = self.net.get_net_class(img_pil)
                        print("nome: ", name)

                        cv2.putText(img, 'Nome: ' + name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        cv2.imshow('frame', img)
                        time.sleep(0.1)
                        if cv2.waitKey(1) & 0xFF == ord(b'q'):
                            break

            except:
                print("error video")

        cv2.destroyAllWindows()
        video.release()
        print("end video")
		'''

    def process_image(self):
        self.import_file()
        print("Dentro process image")

        if self.path != "wrong path":
            img = Image.open(self.path).convert("RGB")
            img = img.resize((self.net.width, self.net.height))

            print("model: ", self.net.pth)
            print("net classification2")
            name, cl = self.net.get_net_class(img)

            d = ImageDraw.Draw(img)
            d.text((10, 10), name, fill=(255, 0, 0))
            img = img.resize((self.rsize_x, self.rsize_y))
            self.show_image(img)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Demo()
    sys.exit(app.exec_())
