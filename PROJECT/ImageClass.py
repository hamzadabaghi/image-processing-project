import math
import cv2
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from skimage.color import *
from PIL import Image


class ImageClass:
    def __init__(self, image):
        self.image = image

    # Binarisation

    def Seuillage(self, s):
        imageX = self.image.copy()
        for i in range(1, imageX.shape[0]):
            for j in range(1, imageX.shape[1]):
                if imageX[i, j] < s:
                    imageX[i, j] = 0
                else:
                    imageX[i, j] = 255
        return imageX

    def Otsu(self):
        pixel_number = self.image.shape[0] * self.image.shape[1]
        mean_weigth = 1.0 / pixel_number
        his, bins = np.histogram(self.image, np.arange(0, 257))
        final_thresh = -1
        final_value = -1
        intensity_arr = np.arange(256)
        # This goes from 1 to 254 uint8 range (Pretty sure wont be those values)
        for t in bins[1:-1]:
            pcb = np.sum(his[:t])
            pcf = np.sum(his[t:])
            Wb = pcb * mean_weigth
            Wf = pcf * mean_weigth

            mub = np.sum(intensity_arr[:t] * his[:t]) / float(pcb)
            muf = np.sum(intensity_arr[t:] * his[t:]) / float(pcf)
            # print mub, muf
            value = Wb * Wf * (mub - muf) ** 2

            if value > final_value:
                final_thresh = t
                final_value = value
        final_img = self.image.copy()
        final_img[self.image > final_thresh] = 255
        final_img[self.image < final_thresh] = 0
        return final_img

        # contour

    def grad(self, seuil):
        imageX = self.image.copy()
        imageY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageX[i, j] = self.image[i, j+1] - self.image[i, j]
                imageY[i, j] = self.image[i+1, j] - self.image[i, j]
        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageXY[i, j] = math.sqrt(
                    imageX[i, j] ** 2 + imageY[i, j] ** 2)
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY

    def Sobel(self, seuil):
        imageX = self.image.copy()
        imageY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageY[i, j] = -self.image[i-1, j-1] - 2*self.image[i, j-1] - self.image[i+1, j-1] \
                    + self.image[i - 1, j + 1] + 2 * \
                    self.image[i, j + 1] + self.image[i + 1, j + 1]
                imageX[i, j] = self.image[i-1, j-1] + 2*self.image[i-1, j] + self.image[i - 1, j + 1]\
                    - self.image[i+1, j-1] - 2 * \
                    self.image[i+1, j] - self.image[i + 1, j + 1]
        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageXY[i, j] = math.sqrt(
                    imageX[i, j] ** 2 + imageY[i, j] ** 2)
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY

    def Laplacien(self, seuil):
        imageXY = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                imageXY[i, j] = -4*self.image[i, j] + self.image[i-1, j] + self.image[i+1, j] \
                    + self.image[i, j - 1] + self.image[i, j + 1]
                if imageXY[i, j] < seuil:
                    imageXY[i, j] = 0
                else:
                    imageXY[i, j] = 255
        return imageXY

    # filtrage

    def Moyenneur(self, taille):
        imagefiltrage = self.image.copy()
        x = int((taille - 1)/2)
        for i in range(x, self.image.shape[0] - x):
            for j in range(x, self.image.shape[1] - x):
                s = 0
                for n in range(-x, x):
                    for m in range(-x, x):
                        s += self.image[i+n, j+m]/(taille*taille)
                imagefiltrage[i, j] = s
                s = 0
        return imagefiltrage

    def Median(self, taille):
        imagefiltrage = self.image.copy()
        x = int((taille - 1) / 2)
        for i in range(x, self.image.shape[0] - x):
            for j in range(x, self.image.shape[1] - x):
                liste = []
                if imagefiltrage[i, j] == 0 or imagefiltrage[i, j] == 255:
                    for n in range(-x, x):
                        for m in range(-x, x):
                            liste.append(imagefiltrage[i + n, j + m])
                    liste.sort()
                    imagefiltrage[i, j] = liste[x + 1]
                    while len(liste) > 0:
                        liste.pop()
        return imagefiltrage

    def h(self, x, y, v):
        x = (1/(2*math.pi*math.pow(v, 2))) * \
            (math.exp(-(math.pow(x, 2)+math.pow(y, 2))/(2*math.pow(v, 2))))
        return x

    def Gaussien(self, v):
        imagefiltrage = self.image.copy()
        x = 1
        for i in range(x, self.image.shape[0] - x):
            for j in range(x, self.image.shape[1] - x):
                s = 0
                for a in range(-x, x):
                    for b in range(-x, x):
                        s = s + self.h(a, b, v)*self.image[i+a, j+b]
                imagefiltrage[i, j] = s
                s = 0
        return imagefiltrage

    # Morphologie

    def dilatation(self, H):
        imagecopy = self.image.copy()
        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                s = 0
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        s = s + self.image[k, l] * H[k - i + 1][l - j + 1]
                if (s == 0):
                    imagecopy[i][j] = 0
                else:
                    imagecopy[i][j] = 255
        return imagecopy

    def Erosion(self, H):
        imagecopy = self.image.copy()

        for i in range(0, self.image.shape[0]):
            for j in range(0, self.image.shape[1]):
                if (self.image[i][j] > 128):
                    self.image[i][j] = 255
                else:
                    self.image[i][j] = 0

        for i in range(1, self.image.shape[0] - 1):
            for j in range(1, self.image.shape[1] - 1):
                s = 0
                for k in range(i - 1, i + 2):
                    for l in range(j - 1, j + 2):
                        s = s + self.image[k, l] * H[k - i + 1][l - j + 1]
                if (s == 2295):
                    imagecopy[i][j] = 255
                else:
                    imagecopy[i][j] = 0
        return imagecopy

    def Ouverture(self, H):
        img = self.Erosion(self.image, H)
        image1 = self.dilatation(img, H)
        return image1

    def Fermeture(self, H):
        img = self.Erosion(self.image, H)
        image1 = self.dilatation(img, H)
        return image1

    # Operations

    def rotate_image(self, angle):
        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (self.image.shape[1], self.image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            self.image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )
        img = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result

    def hist(self):

        k = 0
        try:
            test = self.image.shape[2]
        except IndexError:
            k = 1
        if k == 1:
            h = ImageClass.histo(self.image)
            plt.subplot(1, 1, 1)
            plt.plot(h)
            plt.show()

        else:
            for i in range(0, 3):
                h = ImageClass.histo(self.image[:, :, i])
                plt.subplot(1, 3, i + 1)
                plt.plot(h)
            plt.show()

    def histo(image):
        h = np.zeros(256)
        s = image.shape
        for j in range(s[0]):
            for i in range(s[1]):
                valeur = image[j, i]
                h[valeur] += 1
        return h

    def imhist(im):
        m, n = im.shape
        h = [0.0] * 256
        for i in range(m):
            for j in range(n):
                h[im[i, j]] += 1
        return np.array(h) / (m * n)

    def cumsum(h):
        return [sum(h[:i + 1]) for i in range(len(h))]

    def histeq(self):
        h = ImageClass.imhist(self.image)

        cdf = np.array(ImageClass.cumsum(h))
        sk = np.uint8(255 * cdf)
        s1, s2 = self.image.shape
        Y = np.zeros_like(self.image)
        for i in range(0, s1):
            for j in range(0, s2):
                Y[i, j] = sk[self.image[i, j]]
        return Y

    def etire(self):
        MaxV = np.max(self.image)
        MinV = np.min(self.image)
        Y = np.zeros_like(self.image)
        m = self.image.shape
        for i in range(m[0]):
            for j in range(m[1]):
                Y[i, j] = (255 / (MaxV - MinV) * self.image[i, j] - MinV)
        return Y

