"""Harris corner response and directional structure tensor diagnostics."""
import numpy as np
import cv2

def synthetic_wedge(size=200):
    image = np.full((size, size, 3), (100, 0, 60), dtype=np.uint8)
    pts = np.array([(100,100),(199,130),(199,199),(130,199)], np.int32)
    cv2.fillPoly(image, [pts], (0,255,0))
    return image

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)/255.0

def structure_tensor(gray, sigma=1.2):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    a = cv2.GaussianBlur(gx*gx, (0,0), sigma)
    b = cv2.GaussianBlur(gx*gy, (0,0), sigma)
    c = cv2.GaussianBlur(gy*gy, (0,0), sigma)
    return a,b,c

def harris_response(a,b,c,k=0.04):
    return (a*c-b*b)-k*(a+c)**2

def directional_energy(a,b,c,angles):
    """E(theta)=u(theta)^T M u(theta), shape H x W x angles."""
    angles=np.asarray(angles)
    co=np.cos(angles)[None,None,:]
    si=np.sin(angles)[None,None,:]
    return a[...,None]*co**2+2*b[...,None]*co*si+c[...,None]*si**2

def eigenvalues(a,b,c):
    tr=a+c
    delta=np.sqrt(np.maximum((a-c)**2+4*b*b,0))
    return (tr+delta)/2,(tr-delta)/2
