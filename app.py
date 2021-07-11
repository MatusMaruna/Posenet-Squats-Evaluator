import tempfile
import uuid
from pathlib import Path

from flask import Flask, render_template, request, send_file, session
from flask_socketio import SocketIO
import tensorflow as tf
from keras.backend import set_session
import threading
import globals
from config import general_config
from keyframes_detector.estimator import KeyFramesDetectorEstimator
from kinect.estimator import KinectEstimator
from posenet2kinect.estimator import Pose2KinectEstimator
from posenet_estimator import PosenetEstimator
from video_processor import VideoProcessor
globals.init()
app = Flask(__name__)
# Restrict large files >10MB
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
socketio = SocketIO(app, pingTimeout=5000)
ALLOWED_EXTENSIONS = ["avi"]

graph = tf.get_default_graph()
sess = tf.Session()
frameQualityThreshold = 0

with graph.as_default():
    set_session(sess)
    pose_estim = PosenetEstimator(101)
    pose2kinect_estim = Pose2KinectEstimator(general_config.posenet2kinect_pipeline)
    kinect3d_estim = KinectEstimator(general_config.kinect_pipeline)
    key_frame_estim = KeyFramesDetectorEstimator(general_config.keyframes_detector_pipeline)
    processor = VideoProcessor(pose_estim, pose2kinect_estim, kinect3d_estim, key_frame_estim)


@app.route("/", methods=["GET"])
def index():
    return render_template("home.html")


@app.route("/eval_video", methods=["GET"])
def eval():
    return render_template("eval_video.html")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[
        1].lower() in ALLOWED_EXTENSIONS


@socketio.on('connect')
def connection():
    print('client connected')


@socketio.on('waiting')
def waiting():
    socketio.emit('processing', globals.progress.get(session['video_id']))
    if globals.progress.get(session['video_id']) == 100:
        globals.progress.pop(session['video_id'])
        if globals.results.get(session['video_id'])["status"] is not "SUCCESS":
            socketio.emit('error', globals.results.get(session['video_id'])["message"])
            return
        frame_quality = globals.results[session['video_id']]["data"]["frame_quality"]
        exercise_performance = globals.results[session['video_id']]["data"]["exercise_performance"]
        if exercise_performance is None:
            socketio.emit('error', globals.results[session['video_id']]["data"]["message"])
            return
        exercise_quality = globals.results[session['video_id']]["data"]["exercise_quality"]
        if exercise_quality is "bad":
            socketio.emit('error', globals.results[session['video_id']]["data"]["message"])
            return
        arr = [frame_quality, exercise_quality, exercise_performance]
        socketio.emit('processed', arr)
        globals.results.pop(session['video_id'])


def evaluate_video(file, video_processor):
    global graph
    global sess
    fileName = str(uuid.uuid1())
    temp_video = tempfile.NamedTemporaryFile(prefix=fileName, suffix=".avi", delete=False)
    temp_video.write(file.read())
    temp_csv = tempfile.NamedTemporaryFile(prefix=fileName, suffix=".csv", delete=False)
    session['data_id'] = temp_csv.name
    session['video_id'] = temp_video.name
    print(temp_video.name)
    with graph.as_default():
        set_session(sess)
        globals.progress[session['video_id']] = 0
        thread = threading.Thread(target=video_processor.process, args=(temp_video.name, temp_csv.name, sess,
                                                                        graph, session['video_id']))
        thread.daemon = True  # Daemonize thread
        thread.start()  # Start the execution
    return render_template("processing_video.html", msg="Processing...")


@app.route("/eval_video", methods=["POST"])
def submit_video():
    global processor
    request_file = request.files['data_file']
    if not request_file:
        return render_template("eval_video.html", msg="Please select a file")
    if not allowed_file(request_file.filename):
        print(request_file.filename)
        return render_template("eval_video.html", msg="Error: You can only upload " + str(
            ALLOWED_EXTENSIONS).strip('[]'))
    return evaluate_video(request_file, processor)


@app.route("/download_data", methods=["GET"])
def download_data():
    if session.get('data_id') is None:
        return render_template("eval_video.html", msg="You need to process a file before downloading")
    if Path(session.get('data_id')).exists:
        return send_file(Path(session.get('data_id')), as_attachment=True, mimetype="text/csv",
                         attachment_filename="data.csv")


@app.route("/download_video", methods=["GET"])
def download_video():
    if session.get('video_id') is None:
        return render_template("eval_video.html", msg="You need to process a file before downloading")
    print(Path(session.get('video_id')).exists())
    if Path(session.get('video_id')).exists:
        return send_file(Path(session.get('video_id')), as_attachment=True, mimetype="video/x-msvideo",
                         attachment_filename="video.avi")



if __name__ == "__main__":
    app.secret_key = 'xGppaORyBtoIEd6eswBr'  # TODO Replace by env variable!
    socketio.run(app, host=general_config.service.host, port=int(general_config.service.port))







