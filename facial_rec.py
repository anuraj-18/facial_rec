
import numpy as np
import cv2
from nn import ANN

def init_train_data():
    X = []
    for i in range(12, 22):
        name = "images/shubham" + str(i) + ".jpg"
        arr = cv2.imread(str(name)) # 640x480x3 array
        arr = np.reshape(arr,(1,480*640*3))/255
        X.append(arr[0])
    for i in range(5):
        name = "images/mahika" + str(i) + ".jpg"
        arr = cv2.imread(name) # 640x480x3 array
        arr = np.reshape(arr, (1, 480*640*3))/255
        X.append(arr[0])
    for i in range(1, 5):
        name = "images/dad" + str(i)+".jpg"
        arr = cv2.imread(name) # 640x480x3 array
        arr = np.reshape(arr, (1,480*640*3))/255
        X.append(arr[0])

    for i in range(5):
        name = "images/mum" + str(i) + ".jpg"
        arr = cv2.imread(name) # 640x480x3 array
        arr = np.reshape(arr, (1,480*640*3))/255
        X.append(arr[0])

    X = np.array(X)
    X = X.T

    Y = []
    for i in range(X.shape[1]):
        if i<10:
            k = [1, 0, 0, 0]
            Y.append(k)
        elif i >= 10 and i < 15:
            k = [0, 1, 0, 0]
            Y.append(k)
        elif i >= 15 and i < 19:
            k = [0, 0, 1, 0]
            Y.append(k)
        else:
            k = [0, 0, 0, 1]
            Y.append(k)
    Y = np.array(Y)
    Y = Y.T
    return X, Y

def create_input_data(video):
    X = []
    check, frame = video.read()
    frame = np.reshape(frame, (1,480*640*3))
    X.append(frame[0])
    X = np.array(X)
    X = X.T
    return X

def testOutput(nn):
    video = cv2.VideoCapture(0)
    d={0:"Shubham", 1:"Mahika", 2:"Raj", 3:"Anupma"}

    while True:
        X = create_input_data(video)
        s = nn.test_output(X)
        m = np.amax(s)
        print(s)
        for i in range(len(s)):
            if s[i][0] == m:
                print("This is " + d[i])
                break

        check, frame = video.read()
        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        
        if key == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()

layer_dims = [480*640*3, 7, 7, 4]
X, Y = init_train_data()
nn = ANN(layer_dims)
nn.train(X,Y)
testOutput(nn)
