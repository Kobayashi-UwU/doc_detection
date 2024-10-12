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
        document.getElementById("downloadButton").disabled = (
            True  # Disable download button
        )
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


def download_picture(e=None):
    # Check if the image is ready to download
    output_image = document.getElementById("output_image")
    if output_image.src:
        link = js.document.createElement("a")
        link.download = "scanned_document.png"  # You can change the filename here
        link.href = output_image.src  # Get the source of the output image
        link.click()
    else:
        console.error("No image available to download.")


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
            process_resized_and_original(img, time.time())

        reader.onload = create_proxy(onload)
        reader.readAsDataURL(blob)

    fetch_image(img_src)


# Update threshold values on slider change
def updateThreshold1(event):
    document.getElementById("threshold1Value").innerText = event.target.value


def updateThreshold2(event):
    document.getElementById("threshold2Value").innerText = event.target.value


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
        process_resized_and_original(img, start_time)

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
            process_resized_and_original(img, start_time)

        reader.onload = create_proxy(onload)
        reader.readAsDataURL(file)


def process_resized_and_original(img, start_time):
    console.log("Processing")
    h, w, c = img.shape
    print("width:  ", w)
    print("height: ", h)
    print("channel:", c)

    # Output image size
    img = cv2.resize(img, (1080, 1920))

    original_height, original_width = img.shape[:2]

    # Step 1: Resize image
    heightImg, widthImg = 240, 180
    resized_img = cv2.resize(img, (widthImg, heightImg))

    # Step 2: Process resized image to find document
    imgGray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

    # Use original threshold values that worked for you
    threshold1 = int(document.getElementById("threshold1").value)
    threshold2 = int(document.getElementById("threshold2").value)
    imgThreshold = cv2.Canny(imgBlur, threshold1, threshold2)

    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgThreshold, kernel, iterations=2)
    imgThreshold = cv2.erode(imgDial, kernel, iterations=1)

    # Debug: Draw contours on resized image
    contours, _ = cv2.findContours(
        imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    debug_img = resized_img.copy()
    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 3)

    # Encode the contour debug image for display
    _, contour_encoded = cv2.imencode(".png", debug_img)
    contour_base64 = base64.b64encode(contour_encoded).decode("utf-8")

    # Fix MIME type for base64 encoded image
    contour_image = document.getElementById("output_contour")
    contour_image.src = f"data:image/png;base64,{contour_base64}"
    contour_image.style.display = "block"

    console.log("No biggest contour found, using bounding box")
    # If no biggest contour, find the bounding rectangle of the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Scale coordinates back to original size
        scale_x = original_width / widthImg
        scale_y = original_height / heightImg
        x = int(x * scale_x)
        y = int(y * scale_y)
        w = int(w * scale_x)
        h = int(h * scale_y)

        # Draw rectangle on the original image
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw the corner points of the rectangle
        corners = [(x, y), (x + w, y), (x, y + h), (x + w, y + h)]
        for corner in corners:
            cv2.circle(img, corner, 5, (0, 0, 255), -1)  # Red points

        # Encode the modified image to display it
        _, img_encoded = cv2.imencode(".png", img)
        img_base64 = base64.b64encode(img_encoded).decode("utf-8")
        output_image = document.getElementById("output_image")
        output_image.src = f"data:image/png;base64,{img_base64}"
        output_image.style.display = "block"
    else:
        console.log("No contours found, unable to process image")

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
