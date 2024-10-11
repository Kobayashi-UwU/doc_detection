from js import document, navigator, window, console, Promise, Object, JSON
from pyodide.ffi import create_proxy

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

    # Display the canvas
    canvas.style.display = "block"

    console.log("Photo taken")


def main():
    get_cameras().then(
        lambda _: console.log("Cameras fetched successfully"),
        lambda err: console.error("Error fetching cameras:", err),
    )

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


main()
