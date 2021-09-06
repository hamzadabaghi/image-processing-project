from ImageClass import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.color import *
from PIL import Image


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1006, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.ProjetTitreLab = QtWidgets.QLabel(self.centralwidget)
        self.ProjetTitreLab.setGeometry(QtCore.QRect(440, 20, 181, 21))
        font = QtGui.QFont()
        font.setFamily("Oswald")
        font.setPointSize(12)
        self.ProjetTitreLab.setFont(font)
        self.ProjetTitreLab.setObjectName("ProjetTitreLab")
        self.copyrightLabel = QtWidgets.QLabel(self.centralwidget)
        self.copyrightLabel.setGeometry(QtCore.QRect(860, 510, 121, 21))
        font = QtGui.QFont()
        font.setFamily("Oswald")
        font.setPointSize(9)
        self.copyrightLabel.setFont(font)
        self.copyrightLabel.setObjectName("copyrightLabel")
        self.imageTraitee = QtWidgets.QLabel(self.centralwidget)
        self.imageTraitee.setGeometry(QtCore.QRect(680, 60, 301, 271))
        self.imageTraitee.setAutoFillBackground(True)
        self.imageTraitee.setText("")
        self.imageTraitee.setObjectName("imageTraitee")
        self.imageTraiteeLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageTraiteeLabel.setGeometry(QtCore.QRect(770, 340, 101, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.imageTraiteeLabel.setFont(font)
        self.imageTraiteeLabel.setObjectName("imageTraiteeLabel")
        self.imageInitiale = QtWidgets.QLabel(self.centralwidget)
        self.imageInitiale.setGeometry(QtCore.QRect(340, 60, 301, 271))
        self.imageInitiale.setAutoFillBackground(True)
        self.imageInitiale.setText("")
        self.imageInitiale.setObjectName("imageInitiale")
        self.imageInitialeLabel = QtWidgets.QLabel(self.centralwidget)
        self.imageInitialeLabel.setGeometry(QtCore.QRect(430, 340, 141, 31))
        font = QtGui.QFont()
        font.setFamily("MS Shell Dlg 2")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.imageInitialeLabel.setFont(font)
        self.imageInitialeLabel.setObjectName("imageInitialeLabel")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(653, 60, 20, 271))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.traitementToolBox = QtWidgets.QToolBox(self.centralwidget)
        self.traitementToolBox.setGeometry(QtCore.QRect(30, 60, 241, 281))
        self.traitementToolBox.setObjectName("traitementToolBox")
        self.analyseElementaireTab = QtWidgets.QWidget()
        self.analyseElementaireTab.setGeometry(QtCore.QRect(0, 0, 241, 92))
        self.analyseElementaireTab.setObjectName("analyseElementaireTab")
        self.negatifButton = QtWidgets.QPushButton(self.analyseElementaireTab)
        self.negatifButton.setGeometry(QtCore.QRect(0, 10, 81, 23))
        self.negatifButton.setObjectName("negatifButton")
        self.histogrammeButton = QtWidgets.QPushButton(
            self.analyseElementaireTab)
        self.histogrammeButton.setGeometry(QtCore.QRect(90, 10, 81, 23))
        self.histogrammeButton.setObjectName("histogrammeButton")
        self.egalisationButton = QtWidgets.QPushButton(
            self.analyseElementaireTab)
        self.egalisationButton.setGeometry(QtCore.QRect(0, 40, 81, 23))
        self.egalisationButton.setObjectName("egalisationButton")
        self.etirementButton = QtWidgets.QPushButton(
            self.analyseElementaireTab)
        self.etirementButton.setGeometry(QtCore.QRect(90, 40, 81, 23))
        self.etirementButton.setObjectName("etirementButton")
        self.traitementToolBox.addItem(self.analyseElementaireTab, "")
        self.binarisationTab = QtWidgets.QWidget()
        self.binarisationTab.setGeometry(QtCore.QRect(0, 0, 241, 92))
        self.binarisationTab.setObjectName("binarisationTab")
        self.seuilBinarisation = QtWidgets.QLineEdit(self.binarisationTab)
        self.seuilBinarisation.setGeometry(QtCore.QRect(80, 10, 113, 20))
        self.seuilBinarisation.setInputMask("")
        self.seuilBinarisation.setText("")
        self.seuilBinarisation.setObjectName("seuilBinarisation")
        self.binarisationManuelleButton = QtWidgets.QPushButton(
            self.binarisationTab)
        self.binarisationManuelleButton.setGeometry(
            QtCore.QRect(0, 10, 71, 23))
        self.binarisationManuelleButton.setObjectName(
            "binarisationManuelleButton")
        self.binarisationOtsuButton = QtWidgets.QPushButton(
            self.binarisationTab)
        self.binarisationOtsuButton.setGeometry(QtCore.QRect(0, 40, 71, 21))
        self.binarisationOtsuButton.setObjectName("binarisationOtsuButton")
        self.traitementToolBox.addItem(self.binarisationTab, "")
        self.filtrageTab = QtWidgets.QWidget()
        self.filtrageTab.setGeometry(QtCore.QRect(0, 0, 241, 92))
        self.filtrageTab.setObjectName("filtrageTab")
        self.Gaussienlabel = QtWidgets.QLabel(self.filtrageTab)
        self.Gaussienlabel.setGeometry(QtCore.QRect(0, 10, 47, 13))
        self.Gaussienlabel.setObjectName("Gaussienlabel")
        self.Moyenneurlabel = QtWidgets.QLabel(self.filtrageTab)
        self.Moyenneurlabel.setGeometry(QtCore.QRect(0, 30, 61, 16))
        self.Moyenneurlabel.setObjectName("Moyenneurlabel")
        self.Medianlabel = QtWidgets.QLabel(self.filtrageTab)
        self.Medianlabel.setGeometry(QtCore.QRect(0, 60, 47, 13))
        self.Medianlabel.setObjectName("Medianlabel")
        self.gaussienButton1 = QtWidgets.QPushButton(self.filtrageTab)
        self.gaussienButton1.setGeometry(QtCore.QRect(70, 0, 51, 23))
        self.gaussienButton1.setObjectName("gaussienButton1")
        self.gaussienButton8 = QtWidgets.QPushButton(self.filtrageTab)
        self.gaussienButton8.setGeometry(QtCore.QRect(130, 0, 51, 23))
        self.gaussienButton8.setObjectName("gaussienButton8")
        self.moyenneurButton3 = QtWidgets.QPushButton(self.filtrageTab)
        self.moyenneurButton3.setGeometry(QtCore.QRect(70, 30, 51, 23))
        self.moyenneurButton3.setObjectName("moyenneurButton3")
        self.moyenneurButton5 = QtWidgets.QPushButton(self.filtrageTab)
        self.moyenneurButton5.setGeometry(QtCore.QRect(130, 30, 51, 23))
        self.moyenneurButton5.setObjectName("moyenneurButton5")
        self.medianButton3 = QtWidgets.QPushButton(self.filtrageTab)
        self.medianButton3.setGeometry(QtCore.QRect(70, 60, 51, 23))
        self.medianButton3.setObjectName("medianButton3")
        self.medianButton5 = QtWidgets.QPushButton(self.filtrageTab)
        self.medianButton5.setGeometry(QtCore.QRect(130, 60, 51, 23))
        self.medianButton5.setObjectName("medianButton5")
        self.traitementToolBox.addItem(self.filtrageTab, "")
        self.contourTab = QtWidgets.QWidget()
        self.contourTab.setGeometry(QtCore.QRect(0, 0, 241, 92))
        self.contourTab.setObjectName("contourTab")
        self.contourSobelButton = QtWidgets.QPushButton(self.contourTab)
        self.contourSobelButton.setGeometry(QtCore.QRect(80, 0, 75, 23))
        self.contourSobelButton.setObjectName("contourSobelButton")
        self.contourGradientButton = QtWidgets.QPushButton(self.contourTab)
        self.contourGradientButton.setGeometry(QtCore.QRect(0, 0, 75, 23))
        self.contourGradientButton.setObjectName("contourGradientButton")
        self.contourLaplacienButton = QtWidgets.QPushButton(self.contourTab)
        self.contourLaplacienButton.setGeometry(QtCore.QRect(0, 60, 75, 23))
        self.contourLaplacienButton.setObjectName("contourLaplacienButton")
        self.traitementToolBox.addItem(self.contourTab, "")
        self.morphologieTab = QtWidgets.QWidget()
        self.morphologieTab.setGeometry(QtCore.QRect(0, 0, 241, 92))
        self.morphologieTab.setObjectName("morphologieTab")
        self.morphologieDilatationButton = QtWidgets.QPushButton(
            self.morphologieTab)
        self.morphologieDilatationButton.setGeometry(
            QtCore.QRect(80, 0, 75, 23))
        self.morphologieDilatationButton.setObjectName(
            "morphologieDilatationButton")
        self.morphologieErosionButton = QtWidgets.QPushButton(
            self.morphologieTab)
        self.morphologieErosionButton.setGeometry(QtCore.QRect(0, 0, 75, 23))
        self.morphologieErosionButton.setObjectName("morphologieErosionButton")
        self.morphologieFermetureButton = QtWidgets.QPushButton(
            self.morphologieTab)
        self.morphologieFermetureButton.setGeometry(
            QtCore.QRect(80, 30, 75, 23))
        self.morphologieFermetureButton.setObjectName(
            "morphologieFermetureButton")
        self.morphologieOuvertureButton = QtWidgets.QPushButton(
            self.morphologieTab)
        self.morphologieOuvertureButton.setGeometry(
            QtCore.QRect(0, 30, 75, 23))
        self.morphologieOuvertureButton.setObjectName(
            "morphologieOuvertureButton")
        self.traitementToolBox.addItem(self.morphologieTab, "")
        self.pourcentageLabel = QtWidgets.QLabel(self.centralwidget)
        self.pourcentageLabel.setGeometry(QtCore.QRect(590, 430, 61, 16))
        self.pourcentageLabel.setObjectName("pourcentageLabel")
        self.pourcentageField = QtWidgets.QLineEdit(self.centralwidget)
        self.pourcentageField.setGeometry(QtCore.QRect(660, 430, 113, 20))
        self.pourcentageField.setObjectName("pourcentageField")
        self.redemensionnerButton = QtWidgets.QPushButton(self.centralwidget)
        self.redemensionnerButton.setGeometry(QtCore.QRect(790, 430, 91, 23))
        self.redemensionnerButton.setObjectName("redemensionnerButton")
        self.angleLabel = QtWidgets.QLabel(self.centralwidget)
        self.angleLabel.setGeometry(QtCore.QRect(320, 430, 31, 16))
        self.angleLabel.setObjectName("angleLabel")
        self.angleField = QtWidgets.QLineEdit(self.centralwidget)
        self.angleField.setGeometry(QtCore.QRect(370, 430, 113, 20))
        self.angleField.setObjectName("angleField")
        self.rotationButton = QtWidgets.QPushButton(self.centralwidget)
        self.rotationButton.setGeometry(QtCore.QRect(500, 430, 75, 23))
        self.rotationButton.setObjectName("rotationButton")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1006, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNew = QtWidgets.QAction(MainWindow)
        self.actionNew.setObjectName("actionNew")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionSave)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.traitementToolBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "Projet Traitement image"))
        self.ProjetTitreLab.setText(_translate(
            "MainWindow", "Projet de traitement d\'image "))
        self.copyrightLabel.setText(_translate(
            "MainWindow", "Copyright : Hamza Dabaghi"))
        self.imageTraiteeLabel.setText(
            _translate("MainWindow", "image traitée"))
        self.imageInitialeLabel.setText(
            _translate("MainWindow", "image initiale"))
        self.negatifButton.setText(_translate("MainWindow", "Negatif"))
        self.histogrammeButton.setText(_translate("MainWindow", "Histogramme"))
        self.egalisationButton.setText(_translate("MainWindow", "Egalisation"))
        self.etirementButton.setText(_translate("MainWindow", "Etirement"))
        self.traitementToolBox.setItemText(self.traitementToolBox.indexOf(
            self.analyseElementaireTab), _translate("MainWindow", "Analyse Elementaire"))
        self.binarisationManuelleButton.setText(
            _translate("MainWindow", "Manuelle"))
        self.binarisationOtsuButton.setText(_translate("MainWindow", "Otsu"))
        self.traitementToolBox.setItemText(self.traitementToolBox.indexOf(
            self.binarisationTab), _translate("MainWindow", "Binarisation"))
        self.Gaussienlabel.setText(_translate("MainWindow", "Gaussien"))
        self.Moyenneurlabel.setText(_translate("MainWindow", "Moyenneur"))
        self.Medianlabel.setText(_translate("MainWindow", "Median"))
        self.gaussienButton1.setText(_translate("MainWindow", "0.1"))
        self.gaussienButton8.setText(_translate("MainWindow", "0.8"))
        self.moyenneurButton3.setText(_translate("MainWindow", "3 * 3"))
        self.moyenneurButton5.setText(_translate("MainWindow", "5 * 5"))
        self.medianButton3.setText(_translate("MainWindow", "3 * 3"))
        self.medianButton5.setText(_translate("MainWindow", "5 * 5"))
        self.traitementToolBox.setItemText(self.traitementToolBox.indexOf(
            self.filtrageTab), _translate("MainWindow", "Filtrage"))
        self.contourSobelButton.setText(_translate("MainWindow", "Sobel"))
        self.contourGradientButton.setText(
            _translate("MainWindow", "Gradient"))

        self.contourLaplacienButton.setText(
            _translate("MainWindow", "Laplacien"))
        self.traitementToolBox.setItemText(self.traitementToolBox.indexOf(
            self.contourTab), _translate("MainWindow", "Contour"))
        self.morphologieDilatationButton.setText(
            _translate("MainWindow", "Dilatation"))
        self.morphologieErosionButton.setText(
            _translate("MainWindow", "Erosion"))
        self.morphologieFermetureButton.setText(
            _translate("MainWindow", "Fermeture"))
        self.morphologieOuvertureButton.setText(
            _translate("MainWindow", "Ouverture"))
        self.traitementToolBox.setItemText(self.traitementToolBox.indexOf(
            self.morphologieTab), _translate("MainWindow", "Morphologie"))
        self.pourcentageLabel.setText(_translate("MainWindow", "Pourcentage"))
        self.redemensionnerButton.setText(
            _translate("MainWindow", "Redemensionner"))
        self.angleLabel.setText(_translate("MainWindow", "Angle"))
        self.rotationButton.setText(_translate("MainWindow", "Rotation"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionNew.setText(_translate("MainWindow", "New"))
        self.actionSave.setText(_translate("MainWindow", "Save"))

        # connect buttons to the functions below :

        self.actionNew.triggered.connect(self.openFile)
        self.actionSave.triggered.connect(self.enregistrementimage)

        self.binarisationOtsuButton.clicked.connect(self.BinarisationOtsu)
        self.binarisationManuelleButton.clicked.connect(self.BinarisationLocal)

        self.negatifButton.clicked.connect(self.negatif)
        self.egalisationButton.clicked.connect(self.egalisation)

        self.etirementButton.clicked.connect(self.etir)

        self.histogrammeButton.clicked.connect(self.histo)

        self.rotationButton.clicked.connect(self.rotate)
        self.redemensionnerButton.clicked.connect(self.redim)

        # filtrage

        self.gaussienButton1.clicked.connect(self.gaussian1)
        self.gaussienButton8.clicked.connect(self.gaussian8)

        self.moyenneurButton3.clicked.connect(self.Moyenneur3)
        self.moyenneurButton5.clicked.connect(self.Moyenneur5)

        self.medianButton3.clicked.connect(self.median3)
        self.medianButton5.clicked.connect(self.median5)

        # contour

        self.contourGradientButton.clicked.connect(self.grad)
        self.contourSobelButton.clicked.connect(self.Sobel)
        self.contourLaplacienButton.clicked.connect(self.laplacien)

        # Morphologie

        self.morphologieErosionButton.clicked.connect(self.Erosion)
        self.morphologieDilatationButton.clicked.connect(self.dilatation)

        self.morphologieOuvertureButton.clicked.connect(self.ouverture)
        self.morphologieFermetureButton.clicked.connect(self.fermeture)

    # openFile method :

    def openFile(self):

        nom_fichier = QFileDialog.getOpenFileName()
        self.path = nom_fichier[0]
        pathx = self.path
        pixmap = QtGui.QPixmap(pathx)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)

        self.imageInitiale.setPixmap(QtGui.QPixmap(pixmap4))

    def enregistrementimage(self):
        fileName = QFileDialog.getSaveFileName(
            None, 'some text', "untitled.png", "Image Files (*.jpg *.gif *.bmp *.png)")
        self.fileName = fileName[0]
        print(fileName[0]+'ssss')
        cv2.imwrite(fileName[0], self.mat)

    def rotate(self):
        anglevalue = int(self.angleField.text())
        image = cv2.imread(self.path)
        o = ImageClass(image)
        img = o.rotate_image(anglevalue)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def redim(self):
        image = cv2.imread(self.path)
        pourcentage = int(self.pourcentageField.text())
        scale_percent = pourcentage
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        img = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def negatif(self):
        image = cv2.imread(self.path)
        img = 255 - image
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def Moyenneur5(self):
        image = cv2.imread(self.path)
        f = ImageClass(image)
        img = f.Moyenneur(5)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def Moyenneur3(self):
        image = cv2.imread(self.path)
        f = ImageClass(image)
        img = f.Moyenneur(3)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def gaussian1(self):
        image = cv2.imread(self.path)
        f = ImageClass(image)
        img = f.Gaussien(0.1)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def gaussian8(self):
        image = cv2.imread(self.path)
        f = ImageClass(image)
        img = f.Gaussien(0.8)
        self.mat = img
        height, width, byteValue = img.shape
        print(byteValue)
        if byteValue == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def median3(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        if byteValue == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = ImageClass(image)
            img = f.Median(3)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            f = ImageClass(image)
            img = f.Median(3)
            self.mat = img
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def median5(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        if byteValue == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = ImageClass(image)
            img = f.Median(5)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            f = ImageClass(image)
            img = f.Median(5)
            self.mat = img
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def grad(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            c = ImageClass(imag)
            img = c.grad(20)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            c = ImageClass(image)
            img = c.grad(20)
            self.mat = img
            imag = QtGui.QImage(
                img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)

        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def Sobel(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            c = ImageClass(imag)
            img = c.Sobel(50)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            c = ImageClass(image)
            img = c.Sobel(50)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)

        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def laplacien(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            c = ImageClass(imag)
            img = c.Laplacien(20)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            c = ImageClass(image)
            img = c.Laplacien(20)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def Erosion(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m = ImageClass(imag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m.Erosion(h)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            m = ImageClass(image)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m.Erosion(h)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def dilatation(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m = ImageClass(imag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m.dilatation(h)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            m = ImageClass(image)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m.dilatation(h)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def ouverture(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m = ImageClass(imag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            imaag = m.Erosion(h)
            m1 = ImageClass(imaag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m1.dilatation(h)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            m = ImageClass(image)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            imaag = m.Erosion(h)
            m1 = ImageClass(imaag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m1.dilatation(h)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def fermeture(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            m = ImageClass(imag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            imaag = m.dilatation(h)
            m1 = ImageClass(imaag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m1.Erosion(h)
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            m = ImageClass(image)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            imaag = m.dilatation(h)
            m1 = ImageClass(imaag)
            h = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
            img = m1.Erosion(h)
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def BinarisationOtsu(self):
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        if byteValue == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            b = ImageClass(image)
            print('hello')
            img = b.Otsu()
            self.mat = img
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(img, width, height, byteValue *
                                width, QtGui.QImage.Format_RGB888)
        else:
            b = ImageClass(image)
            img = b.Otsu()
            self.mat = img
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def BinarisationLocal(self):
        name = int(self.seuilBinarisation.text())
        image = cv2.imread(self.path)
        height, width, byteValue = image.shape
        print(byteValue)
        if byteValue == 3:
            imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f = ImageClass(imag)
            img = f.Seuillage(name)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            imag = QtGui.QImage(
                img, width, height, byteValue * width, QtGui.QImage.Format_RGB888)
        else:
            f = ImageClass(image)
            img = f.Seuillage(name)
            imag = QtGui.QImage(
                img, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(381, 341)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def histo(self):
        image = cv2.imread(self.path)
        o = ImageClass(image)
        o.hist()

    def egalisation(self):
        image = cv2.imread(self.path)
        imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        o = ImageClass(imag)
        img = o.histeq()
        self.mat = img
        imag = QtGui.QImage(
            img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))

    def etir(self):
        image = cv2.imread(self.path)
        imag = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        o = ImageClass(imag)
        img = o.etire()
        self.mat = img
        imag = QtGui.QImage(
            img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_Grayscale8)
        pixmap = QtGui.QPixmap(imag)
        pixmap4 = pixmap.scaled(351, 341, QtCore.Qt.KeepAspectRatio)
        self.imageTraitee.setPixmap(QtGui.QPixmap(pixmap4))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
