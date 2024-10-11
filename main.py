import js
from pyodide.ffi import create_proxy
import cv2
import numpy as np
from js import document, FileReader, navigator, window, console, Promise, Object, JSON
import base64
import time

canvas = js.document.querySelector("#canvas")
process_button = js.document.querySelector("#process-button")
image_input = js.document.querySelector("#image-input")
time_display = js.document.querySelector("#time-display")

stream = None


def get_cameras():
    def wrapped(resolve, reject):
        def on_devices(devices):
            video_devices = [
                device for device in devices if device.kind == "videoinput"
            ]
            camera_select = document.getElementById("cameraSelect")
            camera_select.innerHTML = ""

            for index, device in enumerate(video_devices):
                option = document.createElement("option")
                option.value = device.deviceId
                option.text = device.label or f"Camera {index + 1}"
                camera_select.appendChild(option)

            if len(video_devices) > 1:
                back_camera = next(
                    (
                        device
                        for device in video_devices
                        if "back" in device.label.lower()
                    ),
                    None,
                )
                if back_camera:
                    camera_select.value = back_camera.deviceId

            console.log(f"Found {len(video_devices)} video devices")
            resolve()

        navigator.mediaDevices.enumerateDevices().then(on_devices).catch(
            lambda err: reject(str(err))
        )

    return Promise.new(create_proxy(wrapped))


def start_camera(event):
    global stream
    if stream:
        stop_camera()

    selected_device_id = document.getElementById("cameraSelect").value
    console.log(f"Selected device ID: {selected_device_id}")

    def on_stream(s):
        global stream
        stream = s
        video = document.getElementById("video")
        video.srcObject = stream
        document.getElementById("startButton").disabled = True
        document.getElementById("stopButton").disabled = False
        document.getElementById("takePhotoButton").disabled = False
        console.log("Camera started successfully")

    def on_error(err):
        console.error("Error accessing camera:", str(err))
        document.getElementById("video").innerHTML = f"Error: {err}"

    constraints = Object.new()
    constraints.video = True
    if selected_device_id:
        constraints.video = Object.new()
        constraints.video.deviceId = Object.new()
        constraints.video.deviceId.exact = selected_device_id

    console.log("Constraints:", JSON.stringify(constraints))

    navigator.mediaDevices.getUserMedia(constraints).then(on_stream).catch(on_error)


def stop_camera(event=None):
    global stream
    if stream:
        for track in stream.getTracks():
            track.stop()
        document.getElementById("video").srcObject = None
        document.getElementById("startButton").disabled = False
        document.getElementById("stopButton").disabled = True
        document.getElementById("takePhotoButton").disabled = True
        console.log("Camera stopped")


def take_photo(event):
    video = document.getElementById("video")
    canvas = document.getElementById("canvas")
    photo = document.getElementById("photo")

    # Set canvas dimensions to match the video
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    # Draw the current video frame on the canvas
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height)

    process_image(None)
    # Display the canvas
    canvas.style.display = "block"

    console.log("Photo taken")


def download_picture(e=None):
    link = js.document.createElement("a")
    link.download = "gotchya.png"
    link.href = document.getElementById("canvas").toDataURL()
    link.click()


def load_library_photo(e):
    img_src = e.target.src  # Get the source of the clicked image
    js.console.log(f"Loading image from: {img_src}")

    def fetch_image(src):
        # Fetch image data from the given URL (src)
        js.fetch(src).then(lambda response: response.blob()).then(process_blob)

    def process_blob(blob):
        reader = FileReader.new()

        def onload(event):
            data_url = reader.result
            base64_data = data_url.split(",")[1]
            img_data = np.frombuffer(
                bytearray(base64.b64decode(base64_data)), dtype=np.uint8
            )
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            process_document(img, time.time())

        reader.onload = create_proxy(onload)
        reader.readAsDataURL(blob)

    fetch_image(img_src)


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


def resize_image(img, width=1080, height=1920):
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


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

        # Resize the image before processing
        resized_img = resize_image(img)
        process_document(
            resized_img, start_time
        )  # Process the captured image from canvas
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

            # Resize the image before processing
            resized_img = resize_image(img)
            process_document(resized_img, start_time)  # Process the uploaded image

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

# Camera
start_button = document.getElementById("startButton")
stop_button = document.getElementById("stopButton")
take_photo_button = document.getElementById("takePhotoButton")

start_camera_proxy = create_proxy(start_camera)
stop_camera_proxy = create_proxy(stop_camera)
take_photo_proxy = create_proxy(take_photo)

start_button.addEventListener("click", start_camera_proxy)
stop_button.addEventListener("click", stop_camera_proxy)
take_photo_button.addEventListener("click", take_photo_proxy)

console.log("Event listeners attached")

# Process button
process_button.addEventListener("click", create_proxy(process_image))

# Threshold bar
threshold1_proxy = create_proxy(updateThreshold1)
threshold2_proxy = create_proxy(updateThreshold2)
document.getElementById("threshold1").addEventListener("input", threshold1_proxy)
document.getElementById("threshold2").addEventListener("input", threshold2_proxy)

# Photo library
library_photos = document.querySelectorAll(".library-photo")
for photo in library_photos:
    photo.addEventListener("click", create_proxy(load_library_photo))

get_cameras().then(
    lambda _: console.log("Cameras fetched successfully"),
    lambda err: console.error("Error fetching cameras:", err),
)
