import js
from pyodide.ffi import create_proxy
import cv2
import numpy as np
from js import document, FileReader
import base64
import time


start_camera_button = js.document.querySelector("#start-camera")
stop_camera_button = js.document.querySelector("#stop-camera")
video = js.document.querySelector("#video")
take_photo_button = js.document.querySelector("#take-photo")
canvas = js.document.querySelector("#canvas")
process_button = js.document.querySelector("#process-button")
image_input = js.document.querySelector("#image-input")
time_display = js.document.querySelector("#time-display")


async def start_camera_click(e):
    media = js.Object.new()
    media.audio = False
    media.video = True
    stream = await js.navigator.mediaDevices.getUserMedia(media)
    video.srcObject = stream


def stop_camera_click(e):
    stream = video.srcObject
    if stream:
        tracks = stream.getTracks()
        for track in tracks:
            track.stop()  # Stop each track (audio, video, etc.)
        video.srcObject = None  # Clear the video stream


def take_photo(e):
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height)
    image_data_url = canvas.toDataURL("image/jpeg")
    js.console.log(image_data_url)


def download_picture(e=None):
    link = js.document.createElement("a")
    link.download = "gotchya.png"
    link.href = document.getElementById("canvas").toDataURL()
    link.click()


# Update threshold values on slider change
def updateThreshold1(event):
    document.getElementById("threshold1Value").innerText = event.target.value


def updateThreshold2(event):
    document.getElementById("threshold2Value").innerText = event.target.value


def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest, max_area


def reorder(points):
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def stackImages(imgArray, scale, labels=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        hor = [np.zeros((height, width, 3), np.uint8)] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        ver = np.hstack(imgArray)
    return ver


def take_photo(e):
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height)
    # Instead of just logging the image data URL, we can process the image immediately
    process_image(None)  # Call process_image directly after taking the photo


def process_image(event):
    start_time = time.time()  # Start timer

    image_input = document.getElementById("image-input")
    file = image_input.files.item(0)

    if not file:
        # No file uploaded, process image from the canvas
        image_data_url = canvas.toDataURL("image/jpeg")
        base64_data = image_data_url.split(",")[1]  # Extract only Base64 part
        img_data = np.frombuffer(
            bytearray(base64.b64decode(base64_data)), dtype=np.uint8
        )
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        process_document(img, start_time)  # Process the captured image from canvas
    else:
        # File uploaded, process the uploaded file
        reader = FileReader.new()

        def onload(event):
            data_url = reader.result
            base64_data = data_url.split(",")[1]
            img_data = np.frombuffer(
                bytearray(base64.b64decode(base64_data)), dtype=np.uint8
            )
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            process_document(img, start_time)  # Process the uploaded image

        reader.onload = create_proxy(onload)
        reader.readAsDataURL(file)


# Make sure to keep this function as it is
def process_document(img, start_time):
    # Get threshold values from sliders
    threshold1 = int(document.getElementById("threshold1").value)
    threshold2 = int(document.getElementById("threshold2").value)

    # Image processing pipeline
    heightImg, widthImg = 640, 480
    img = cv2.resize(img, (widthImg, heightImg))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgThreshold = cv2.Canny(imgBlur, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    contours, _ = cv2.findContours(
        imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    biggest, _ = biggestContour(contours)

    if biggest.size != 0:
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32(
            [[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]]
        )
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))

        imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
        imgAdaptiveThre = cv2.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv2.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre = cv2.medianBlur(imgAdaptiveThre, 3)

        stackedImage = stackImages(
            [
                [img, imgGray, imgThreshold],
                [imgWarpColored, imgWarpGray, imgAdaptiveThre],
            ],
            0.75,
        )

        _, img_encoded = cv2.imencode(".png", stackedImage)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")
        output_image = document.getElementById("output_image")
        output_image.src = f"data:image/png;base64,{img_base64}"
        output_image.style.display = "block"

    end_time = time.time()  # End timer
    processing_time = end_time - start_time  # Calculate total processing time
    document.getElementById("time-display").innerText = (
        f"Processing Time: {processing_time:.2f} seconds"
    )


### Event Listener #################################################################

# Open camera button
start_camera_button.addEventListener("click", create_proxy(start_camera_click))

# Close camera button
stop_camera_button.addEventListener("click", create_proxy(stop_camera_click))

# Take photo button
take_photo_button.addEventListener("click", create_proxy(take_photo))

# Process button
process_button.addEventListener("click", create_proxy(process_image))

# Threshold bar
threshold1_proxy = create_proxy(updateThreshold1)
threshold2_proxy = create_proxy(updateThreshold2)
document.getElementById("threshold1").addEventListener("input", threshold1_proxy)
document.getElementById("threshold2").addEventListener("input", threshold2_proxy)
