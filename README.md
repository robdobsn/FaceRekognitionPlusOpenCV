# Facial Recognition using OpenCV and Amazon Rekognition

> Rob Dobson 2018

A front-door face recognition system. A daemon watches a video stream, detects faces locally with OpenCV, checks them against a known set of faces with Amazon Rekognition, and opens the door through the door controller's HTTP API when the match is a known user.

See [this blog post for more details](https://robdobson.com/2018/10/a-face-that-opens-doors/).

## Requirements

- **Python 3.12 or newer.** The pinned dependencies need it (numpy 2.5 needs 3.12+, Pillow 13 needs 3.11+). Older Python versions will fail to install the requirements.
- An AWS account with credentials set up for boto3 (for example with `aws configure` or the `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` environment variables). Both tools use the `eu-west-1` region.
- A video source OpenCV can open, such as an RTSP URL from an IP camera.
- A door controller with an HTTP API (see [Door controller API](#door-controller-api)).
- A display, because the daemon shows the camera feed in an OpenCV window.

## Repository layout

| Folder | Contents |
|---|---|
| [AdminTools/](AdminTools/) | `SetupFacesOnAmazon.py`: uploads the known faces to AWS |
| [Deamon/](Deamon/) | `FacialRecDaemon.py`: the recognition daemon, plus the OpenCV face detector model in `faceDetector/` |

Each folder has its own `requirements.txt`.

## Installation

Create a virtual environment with Python 3.12+ and install the requirements for the tool you want to run:

```sh
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r Deamon/requirements.txt
pip install -r AdminTools/requirements.txt
```

## 1. Set up the known faces on AWS

Put photos of each person in a folder named after them. The folder names must match the user names in the door controller.

```
FacialData/
  Alice Smith/
    photo1.jpg
    photo2.jpg
  Bob Jones/
    photo1.jpg
```

Then run:

```sh
cd AdminTools
python SetupFacesOnAmazon.py --facesFolder ../FacialData
```

This **replaces** any existing face data. It clears the S3 bucket `facetrackdbimages`, recreates the DynamoDB table and the Rekognition collection (both named `family_collection`), then uploads and indexes every `.jpg`. The S3 bucket must already exist. The DynamoDB table and the Rekognition collection are created if they don't exist.

## 2. Run the daemon

Run the daemon from the `Deamon` folder, because it loads the face detector model from `faceDetector/` relative to the current directory.

Settings are read from a `config.ini` in the current directory (it is git-ignored):

```ini
[DEFAULT]
videoSource = rtsp://camera-address/stream
frontDoorURL = http://door-controller-address
faceCaptureFolder = faceCaptures
```

Command-line arguments override the values in `config.ini`:

```sh
cd Deamon
python FacialRecDaemon.py --videoSource rtsp://camera-address/stream \
    --frontDoorURL http://door-controller-address \
    --faceCaptureFolder faceCaptures
```

| Setting | Meaning |
|---|---|
| `videoSource` | Any source `cv2.VideoCapture` accepts |
| `frontDoorURL` | Base URL of the door controller |
| `faceCaptureFolder` | Folder where detected face images are saved, as a visual log |

Frames are rotated by 270° before detection, to suit the original camera mounting. Rekognition requests are rate-limited to one every 3 seconds, and the door won't be reopened within 15 seconds of the last opening.

## Door controller API

The daemon expects the door controller to provide:

- `GET <frontDoorURL>/users`: a JSON list of users, each with `name`, `enable` and `pin` fields
- `GET <frontDoorURL>/u//<userIndex>/<pin>`: opens the door for that user

A recognised person opens the door only if their name matches an enabled user.
