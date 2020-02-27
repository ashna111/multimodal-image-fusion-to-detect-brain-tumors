import os
from datetime import datetime
from flask import Flask, render_template, request, url_for
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
import cv2
import imageio
import scipy.ndimage as ndi

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))


# Our methods
def convertToIntList(arr):
    result=[]
    for q in arr.strip(']][[').split('],['):
        x=[]
        for i in q.split(','):
            x.append(int(i,10))
        result.append(x)
    return result

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y    
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling 
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d       
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform   
        a dict specifying the rotation, translation and scaling that
        maps X --> Y

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2
        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)
    #rot =1
    #scale=2
    #translate=3
    #transformation values 
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform


# Routes
@app.route("/")
def index():
    return render_template("form.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'static/')
    if not os.path.isdir(target):
        os.mkdir(target)

    mri_file=request.files['mri']
    ct_file=request.files['ct']

    destination1 = "/".join([target, "mri.jpg"])
    mri_file.save(destination1)

    destination2 = "/".join([target, "ct.jpg"])
    ct_file.save(destination2)
    
    points = request.form["points"] #no of points

    return render_template("registration.html", points=points)

@app.route("/register",methods=['POST'])
def register():
    global mriCoord, ctCoord
    mriCoord=convertToIntList(request.form['mriCoord'])
    ctCoord=convertToIntList(request.form['ctCoord'])

    # Registration notebook code
    ct = cv2.imread('static/ct.jpg', 0)
    mri = cv2.imread('static/mri.jpg', 0)
    X_pts = np.asarray(mriCoord)
    Y_pts = np.asarray(ctCoord)
    print(X_pts)
    print(Y_pts)

    d,Z_pts,Tform = procrustes(X_pts,Y_pts)
    R = np.eye(3)
    R[0:2,0:2] = Tform['rotation']

    S = np.eye(3) * Tform['scale'] 
    S[2,2] = 1
    t = np.eye(3)
    t[0:2,2] = Tform['translation']
    M = np.dot(np.dot(R,S),t.T).T
    tr_Y_img = cv2.warpAffine(mri,M[0:2,:],(500,500))
    cv2.imwrite("static/mri_registered.jpg", tr_Y_img)

    # aY_pts = np.hstack((Y_pts,np.array(([[1,1,1,1,1]])).T))
    # tr_Y_pts = np.dot(M,aY_pts.T).T 

    # plt.figure() 
    # plt.subplot(1,3,1)
    # plt.imshow(ct,cmap=cm.gray)
    # plt.plot(X_pts[:,0],X_pts[:,1],'bo',markersize=5)
    # plt.subplot(1,3,3)
    # plt.imshow(tr_Y_img,cmap=cm.gray)
    # plt.plot(tr_Y_pts[:,0],tr_Y_pts[:,1],'gx',markersize=5)
    # plt.show()


    return "something"

@app.route("/registerimage")
def registerimage():
    return render_template("imageregistration.html")

@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

if __name__ == "__main__":
    app.run(debug=True)