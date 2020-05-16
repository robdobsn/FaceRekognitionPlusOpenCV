# Daemon runs in the background, detects faces and opens doors
# Rob Dobson 2018

# Grabs frames from a video source
# Detects faces in the frames using code from https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
# Validates faces with Amazon Rekognition
# If probability of match high enough then find if the door should be opened for this person

import os
# import win32file
# import win32con
import boto3
import io
from PIL import Image
import requests
import time
import logging
import cv2
import imutils
import numpy as np
import datetime
import argparse
import configparser

####################################
# Door control
####################################

class DoorController:

    def __init__(self, doorUrl):
        self.doorUrl = doorUrl
        self.lastDoorOpenTime = None
        self.minTimeBetweenOpens = 15

    def doorAlreadyOpen(self):
        return not ((self.lastDoorOpenTime is None) or (time.time() - self.lastDoorOpenTime > self.minTimeBetweenOpens))

    def getUsers(self):
        try:
            r = requests.get(self.doorUrl + "/users")
            self.users = r.json()
        except Exception as excp:
            logging.error(f"getUsers: Failed to get users from door, {str(excp)}")
            return False
        # logging.debug(self.users)
        return True

    def getUserIdx(self, userName):
        for userIdx, user in enumerate(self.users):
            userEnable = user.get("enable", "0")
            if int(userEnable) != 0:
                name = user.get("name", "")
                if name == userName:
                    return userIdx
        return -1

    def openDoorIfUserValid(self, userName):
        userIdx = self.getUserIdx(userName)
        if userIdx == -1:
            logging.info(f"openDoorIfUserValid: User Is Not Valid, username = {userName}")
            return False
        try:
            r = requests.get(self.doorUrl + "/u//" + str(userIdx) + "/" + self.users[userIdx].get("pin",""))
            logging.info(f"openDoorIfUserValid: User Is Valid - Door opening for {userName}")
            self.lastDoorOpenTime = time.time()
        except Exception as excp:
            logging.warning(f"openDoorIfUserValid: Failed to open door using door API, {str(excp)}")
        return True

####################################
# Grab faces from video stream
####################################

class FaceGrabber():

    def __init__(self, videoSource, faceDetector, callback):
        self.videoSource = videoSource
        self.faceDetector = faceDetector
        self.callback = callback

    def start(self):

        # Start collection
        logging.info("Starting video stream...")
        vs = cv2.VideoCapture(VIDEO_SOURCE)
        # vs = cv2.VideoCapture(0)
        time.sleep(2.0)

        # Loop over frames from the video file stream
        lastFaceMatchResult = None
        while True:
            # grab the frame from the threaded video stream
            frame = vs.read()[1]

            # resize the frame to have a width of 800 pixels (while
            # maintaining the aspect ratio), and then grab the image
            # dimensions
            # print("Original image h,w", frame.shape[:2])
            frame = imutils.resize(frame, width=800)
            frame = imutils.rotate_bound(frame, 270)
            (h, w) = frame.shape[:2]
            # print("Resized image h,w", h, w)

            # construct a blob from the image
            imageBlob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            # apply OpenCV's deep learning-based face detector to localize
            # faces in the input image
            self.faceDetector.detector.setInput(imageBlob)
            detections = self.faceDetector.detector.forward()

            # loop over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e., probability) associated with
                # the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > FACE_DETECT_CONFIDENCE_LEVEL:
                    # compute the (x, y)-coordinates of the bounding box for
                    # the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    startX = max(startX-20, 0)
                    startY = max(startY-20, 0)
                    endX = min(endX+10, w-1)
                    endY = min(endY+10, h-1)

                    # # Crop image
                    # image_crop = frame.crop((startX, startY, endX, endY))
                    # (fH, fW) = image_crop.shape[:2]
                    #
                    # # ensure the face width and height are sufficiently large
                    # if fW < 20 or fH < 20:
                    #     continue

                    # extract the face ROI
                    face = frame[startY:endY, startX:endX].copy()
                    (fH, fW) = face.shape[:2]

                    # ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # construct a blob for the face ROI and callback
                    # faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                    #                                  (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    # self.callback(faceBlob)
                    person = self.callback(face)

                    # draw the bounding box of the face along with the
                    # associated probability
                    if person != "__REPEAT__":
                        lastFaceMatchResult = person

                    # Label the frame
                    text = "{} Face Confidence {:.2f}%".format(lastFaceMatchResult if lastFaceMatchResult else "Unknown", confidence * 100)
                    y = startY - 10 if startY - 10 > 10 else startY + 10
                    cv2.rectangle(frame, (startX, startY), (endX, endY),
                                  (0, 0, 255), 2)
                    cv2.putText(frame, text, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            # show the output frame
            cv2.imshow("Front Door Face Detector", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q") or key == 27:
                break

#############################################
# Face detector using caffe model
#############################################

class FaceDetector():

    def __init__(self):
        # load our serialized face detector from disk
        logging.info("Loading face detector...")
        protoPath = os.path.sep.join(["faceDetector", "deploy.prototxt"])
        modelPath = os.path.sep.join(["faceDetector", "res10_300x300_ssd_iter_140000.caffemodel"])
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

#############################################
# Face recognizer using Amazon Rekognition
#############################################

class FaceRecogniser():

    def __init__(self):
        self.timeOfLastRekognitionRequest = None

    def readyForRequest(self):

        # Check we don't exceed rate of requests
        if self.timeOfLastRekognitionRequest is None:
            return True
        if time.time() - self.timeOfLastRekognitionRequest > MIN_SECS_BETWEEN_REKOGNITION_REQS:
            return True
        return False

    def renameFaceFile(self, origName, person):
        try:
            uniqNum = 0
            dateTimeStr = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
            while True:
                testName = f"{FACE_CAPTURE_FOLDER}/{dateTimeStr}_{person}_{uniqNum}.jpg"
                if os.path.isfile(testName):
                    uniqNum += 1
                    if uniqNum > 100000:
                        return False
                    continue
                os.rename(origName, testName)
                break
            return True
        except Exception as excp:
            logging.error(f"RenameFaceFile: {str(excp)}")
            return False

    def recogniseFaces(self, frameWithFace):

        # Requesting now
        self.timeOfLastRekognitionRequest = time.time()

        #
        # logging.info(f"recogniseFaces: image = {imageName}")

        # try:
        #     image = Image.open("../../FacialData/People/Rob Dobson/Rob1.jpg")
        #     stream = io.BytesIO()
        #     image.save(stream, format="JPEG")
        #     image_binary = stream.getvalue()
        # except Exception as excp:
        #     logging.warning(f"recogniseFaces: Can't open {imageName}, {str(excp)}")
        #     return

        # jpegEnc = cv2.imencode(".jpeg", frameWithFace)
        # jpegBytes = bytearray(jpegEnc[1])
        # frameEnc = base64.b64encode(jpegBytes)

        try:
            newFaceName = FACE_CAPTURE_FOLDER + "/newface.jpeg"
            cv2.imwrite(newFaceName, frameWithFace)
            image = Image.open(newFaceName)
            stream = io.BytesIO()
            image.save(stream, format="JPEG")
            image.close()
            image_crop_binary = stream.getvalue()
            # stream = io.BytesIO()
            # image_binary = stream.getvalue()
        except Exception as excp:
            logging.warning(f"recogniseFaces: Can't open {str(excp)}")
            return None

        try:
            # Submit individually cropped image to Amazon Rekognition
            response = rekognition.search_faces_by_image(
                CollectionId='family_collection',
                Image={'Bytes': image_crop_binary}
            )
        except rekognition.exceptions.InvalidParameterException as excp:
            logging.warning(f"recogniseFaces: InvalidParameterException searching for faces ..., {str(excp)}")
            self.renameFaceFile(newFaceName, "InvalidParam")
            return None
        except Exception as excp:
            logging.warning(f"recogniseFaces: Exception searching for faces ..., {str(excp)}")
            self.renameFaceFile(newFaceName, "OtherExcp")
            return None

        person = "ZeroMatches"
        if len(response['FaceMatches']) > 0:
            person = "UnknownMatch"
            # Return results
            # logging.debug('Coordinates ', box)
            for match in response['FaceMatches']:

                face = dynamodb.get_item(
                    TableName='family_collection',
                    Key={'RekognitionId': {'S': match['Face']['FaceId']}}
                )

                if 'Item' in face:
                    person = face['Item']['FullName']['S']
                    if doorController.openDoorIfUserValid(person):
                        break

            self.renameFaceFile(newFaceName, person)

            logging.info(f"recogniseFaces: faceId {match['Face']['FaceId']} confidence {match['Face']['Confidence']} name {person}")
        else:
            logging.info(f"recogniseFaces: no matching faces found")
            self.renameFaceFile(newFaceName, person)

        return person

#############################################
# Callback from face grabber with a face
#############################################

def interestingFrameCallback(frameInfo):

    # Check we don't exceed the maximum rate of Rekognition requests
    if not faceRecogniser.readyForRequest():
        return "__REPEAT__"

    # Make request
    return faceRecogniser.recogniseFaces(frameInfo)

#############################################
# Main
#############################################

if __name__ == "__main__":

    # Setup logging
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

    DEBUG_STORE_FACES = True
    FACE_DETECT_CONFIDENCE_LEVEL = 0.5
    MIN_SECS_BETWEEN_REKOGNITION_REQS = 3

    # Config
    config = configparser.ConfigParser()
    filesRead = config.read("config.ini")
    logging.info(f"Read config file {filesRead}")
    VIDEO_SOURCE = config["DEFAULT"]["videoSource"]
    FRONT_DOOR_URL = config["DEFAULT"]["frontDoorURL"]
    FACE_CAPTURE_FOLDER = config["DEFAULT"]["faceCaptureFolder"]

    # Arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--videoSource", help="URL for video - must be valid in opencv", required=False)
    ap.add_argument("--frontDoorURL", help="URL for front door", required=False)
    ap.add_argument("--faceCaptureFolder", help="folder for facial captures - a visual log", required=False)
    args = vars(ap.parse_args())

    if ("videoSource" in args) and (args["videoSource"] is not None):
        VIDEO_SOURCE = args["videoSource"]
    if ("frontDoorURL" in args) and (args["frontDoorURL"] is not None):
        FRONT_DOOR_URL = args["frontDoorURL"]
    if ("faceCaptureFolder" in args) and (args["faceCaptureFolder"] is not None):
        FACE_CAPTURE_FOLDER = args["faceCaptureFolder"]

    logging.info(f"VideoSource {VIDEO_SOURCE}")
    logging.info(f"FrontDoorURL {FRONT_DOOR_URL}")
    logging.info(f"FaceCaptureFolder {FACE_CAPTURE_FOLDER}")

    # Setup Amazon services
    rekognition = boto3.client('rekognition', region_name='eu-west-1')
    dynamodb = boto3.client('dynamodb', region_name='eu-west-1')

    # Face detector and recogniser
    faceDetector = FaceDetector()
    faceRecogniser = FaceRecogniser()

    # Door controller
    doorController = DoorController(FRONT_DOOR_URL)
    if not doorController.getUsers():
        exit(1)

    # Start face grabber
    faceGrabber = FaceGrabber(0, faceDetector, interestingFrameCallback)
    logging.info("Facial recognition door controller started")
    faceGrabber.start()

