import cv2,os

# Import numpy for matrices calculations
import numpy as np
import time
from pyfingerprint.pyfingerprint import PyFingerprint



#fingerprint module


try:
    f = PyFingerprint('/dev/ttyUSB0', 9600, 0xFFFFFFFF, 0x00000000)

    if ( f.verifyPassword() == False ):
        raise ValueError('The given fingerprint sensor password is wrong!')

except Exception as e:
    print('Exception message: ' + str(e))
    exit(1)

def searchFinger():
    try:
        print('Waiting for finger...')
        time.sleep(2)
        
        while( f.readImage() == False ):
            pass
            time.sleep(3)
            return

        f.convertImage(0x01)
        result = f.searchTemplate()
        positionNumber = result[0]
        accuracyScore = result[1]
        if positionNumber == -1 :
            print('No match found!')
            time.sleep(2)
            return
        else:
            print('Found template at position #' + str(positionNumber))
            time.sleep(2)

    except Exception as e:
        print('Operation failed!')
        print('Exception message: ' + str(e))
        exit(1)


#face recognition
        
def face_rec():
    for c in range(1,100):
        # Create Local Binary Patterns Histograms for face recognization
        recognizer = cv2.createLBPHFaceRecognizer()

        # Load the trained mode
        recognizer.load('trainer39.yml')
        fn_dir = '/home/pi/Desktop/EDD/dataset'
        i=0
        j=0
        k=0
        # Load prebuilt model for Frontal Face
        cascadePath = "haarcascade_frontalface_default.xml"
        (im_width, im_height) = (112, 92)
        # Part 2: Use fisherRecognizer on camera stream
        (images, lables, names, id) = ([], [], {}, 0)
        for (subdirs, dirs, files) in os.walk(fn_dir):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(fn_dir, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    lable = id
                    images.append(cv2.imread(path, 0))
                    lables.append(int(lable))
                id += 1

        face_cascade = cv2.CascadeClassifier(cascadePath)
        webcam = cv2.VideoCapture(0)
        while(i<=50 or j<=50 or k<=50):
            (_, im) = webcam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (im_width, im_height))
                # Try to recognize the face
                prediction = recognizer.predict(gray[y:y+h,x:x+w])
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if prediction[1]<300:
                    if(prediction[0]<50):
                        if(names[prediction[0]]=='sourabh'):
                            cv2.putText(im,'%s - %.0f' % ('sourabh',prediction[1]),(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                            i=i+1
                            
                        elif(names[prediction[0]]=='shubham'):
                            cv2.putText(im,'%s - %.0f' % ('shubham',prediction[1]),(x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
                            j=j+1
                            
                        elif (names[prediction[0]] == 'swapnil'):
                            cv2.putText(im, '%s - %.0f' % ('swapnil', prediction[1]), (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,
                                        (0, 255, 0))
                            k = k + 1
                            
                        else:
                            cv2.putText(im,'not recognized',(x-10, y-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            #print(prediction[0])
            cv2.imshow('OpenCV', im)
            key = cv2.waitKey(10)
        
    webcam.release()
    cv2.destroyAllWindows()
    return prediction[0]

if __name__== "__main__":
    while(1):
        #face recognition
        fid=face_rec()
        print(fid)
        pos=searchFinger()
        print(pos)

        #matching data
        if(fid==1 and pos == 2):
            print('Welcome Sourabh')
            print('Unlocking Lock')
        elif(fid==2 and pos ==3):
            print('Welcome Swapnil')
            print('Unlocking Lock')
        elif(fid==3 and pos ==1):
            print('Welcome Shubham')
            print('Unlocking Lock')
        
