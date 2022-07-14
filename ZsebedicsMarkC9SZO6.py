import cv2
import numpy as np
import math

def add_point_noise(img_in, percentage, value):
    noise = np.copy(img_in)
    n = int(img_in.shape[0] * img_in.shape[1] * percentage)
    for k in range(1, n):
        i = np.random.randint(0, img_in.shape[1])
        j = np.random.randint(0, img_in.shape[0])

        if img_in.ndim == 2:
            noise[j, i] = value

        if img_in.ndim == 3:
            noise[j, i] = [value, value, value]
    return noise

def add_salt_and_pepper_noise(img_in, percentage1, percentage2):
    n = add_point_noise(img_in, percentage1, 255)  # Só
    n2 = add_point_noise(n, percentage2, 0)  # Bors
    cv2.imshow('Borsos',n2)
    return n2



def addNoise(img):
    noise = np.zeros(img.shape[:2], np.int16)
    cv2.randn(noise, 0.0, 20.0)
    imnoise1 = cv2.add(img, noise, dtype=cv2.CV_8UC1)
    cv2.imshow('add', imnoise1)
    imnoise2 = cv2.add(img, noise, dtype=cv2.CV_16SC1)
    imnoisenorm = cv2.normalize(imnoise2, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    cv2.imshow('norm', imnoisenorm)
    return (imnoise1,imnoisenorm)


def rotate(img, rot):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    M = cv2.getRotationMatrix2D(center, rot, 1)
    abs_cos = abs(M[0, 0])
    abs_sin = abs(M[0, 1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)
    M[0, 2] += bound_w / 2 - center[0]
    M[1, 2] += bound_h / 2 - center[1]

    dst = cv2.warpAffine(img, M, dsize=(bound_w, bound_h))
    cv2.imshow('rotate', dst)
    return dst

def lineDetector(image,og_image):
    dst = cv2.Canny(image, 70, 170, None, 3)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLines(dst, 1, np.pi / 189,70)

    sizemax = math.sqrt(cdst.shape[0] ** 2 + cdst.shape[1] ** 2)
    if lines is not None:
        for i in range(0, 1):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + sizemax * (-b)), int(y0 + sizemax * a))
            pt2 = (int(x0 - sizemax * (-b)), int(y0 - sizemax * a))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('vonal',cdst)
            print(('Vonal szoge: ',theta*(180/math.pi)))
            return rotate(og_image ,90+(theta*(180/math.pi)))

def pottyosAzIgazi(domino):
    src = domino.copy()
    Pottyszam = 0

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (5, 5),0.9)

    params = cv2.SimpleBlobDetector_Params()

    params.filterByArea = True
    params.minArea = 100

    params.filterByCircularity = True
    params.minCircularity = 0.7

    params.filterByConvexity = True
    params.minConvexity = 0.3

    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(src)

    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(src, keypoints, blank, (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    number_of_blobs = len(keypoints)
    print(f'Pottyok szama: {number_of_blobs}')

    cv2.imshow('Elipszis Detektalas', blobs)

    detected_circles = cv2.HoughCircles(gray_blurred,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=1, maxRadius=45)

    if detected_circles is not None:

        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            cv2.circle(domino, (a, b), r, (0, 255, 0), 2)

            cv2.circle(domino, (a, b), 1, (0, 0, 255), 3)
            cv2.imshow('Hough kordetektalas', domino)
    else:
        print('A hough korkereso nem talalt kort')
    return Pottyszam

def korvonal(image, ablakok=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    im_floodfill = thresh.copy()
    h, w = thresh.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = thresh | im_floodfill_inv
    cannyd = cv2.Canny(im_out, 10, 30)
    szinesHatar = cv2.cvtColor(cannyd, cv2.COLOR_GRAY2BGR)
    szinesHatar = cv2.multiply(szinesHatar, 1.9)
    kernel = np.ones((3, 3), np.uint8)
    morphed = cv2.dilate(szinesHatar, kernel)
    ored = cv2.bitwise_or(szinesHatar, image)
    duplanOred = cv2.bitwise_or(morphed, image )
    if ablakok:
        cv2.imshow('HatarAlap', im_out)
        cv2.imshow("Hatarok", ored)
        cv2.imshow("Megvastagitott hatarok hogy jobban latszodjon", duplanOred)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return im_out

img = cv2.imread('img/domino-65136_960_720.jpg')  #eredeti kep, erre lett behangolva
#img = cv2.imread('img/domino4.png')              #nagyon egyszeru kep
#img = cv2.imread('img/domino3.jpg')              #lehetne rosszabb
#img = cv2.imread('img/domino5.jpg')              #Ennel aligha van rosszabb

#zajkeltes
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
noise1,noise2 = addNoise(grey)
noise3 = add_salt_and_pepper_noise(img,0.01,0.01)
noise4 = add_salt_and_pepper_noise(img,0.06,0.06)
noise1 = cv2.cvtColor(noise1,cv2.COLOR_GRAY2BGR)
noise2 = cv2.cvtColor(noise2,cv2.COLOR_GRAY2BGR)

cv2.waitKey()
cv2.destroyAllWindows()
print('Zajmentes szegmentalas')
im_out_zajmentes = korvonal(img,True)
print('Gyengen zajos szegmentalas')
im_out_simazaj1 = korvonal(noise1,True)
print('Erosen zajos szegmentalas')
im_out_simazaj2 = korvonal(noise2,True)
print('Gyengen borsos szegmentalas')
im_out_soborsGyenge = korvonal(noise3,True)
print('Erosen Borsos szegmentalas')
im_out_soborsEros = korvonal(noise4,True)

output = cv2.connectedComponentsWithStats(
    im_out_soborsEros, 4, cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

for i in range(1, numLabels):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    if area > 100:
        (cX, cY) = centroids[i]
        output = img.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        componentMask = (labels == i).astype("uint8") * 255
        cv2.imshow("Korbeolelo doboz", output)
        cv2.imshow("Kiszegmentalt alak", componentMask)
        domiImage = img[y:y+h, x:x+w]
        cv2.imshow('domino',domiImage)
        gray = cv2.cvtColor(domiImage, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cannyd = cv2.Canny(thresh, 10, 30)
        cv2.imshow('cannyd', cannyd)
        celkep = None
        try:
            celkep = lineDetector(cannyd,domiImage)
        except Exception:
            print('VonalDetektalasi hiba, megyunk tovabb')
        if celkep is None:
            continue
        #cv2.imwrite(f'dominoReszlet{i}.jpg', celkep)  #ha ki kene irni egyesevel a dominokat
        pottyosAzIgazi(celkep)
        cv2.waitKey()
        cv2.destroyAllWindows()

cv2.waitKey(0)
