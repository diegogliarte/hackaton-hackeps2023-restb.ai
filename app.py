import os

from flask import Flask, request, redirect, render_template

from image import image_to_base64
from main import get_best_photos

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return redirect("/upload-video")


@app.route('/upload-video', methods=['GET'])
def form():
    return render_template('upload_form.html')


@app.route('/upload-video', methods=['POST'])
def upload_video():
    video = request.files['video']
    video_path = os.path.join('uploads', video.filename)
    video.save(video_path)

    best_photos, average_score = get_best_photos(video_path, dev=True)
    best_photos_base64 = [image_to_base64(photo) for photo in best_photos]

    return render_template('image_display.html', images=best_photos_base64, average_score=average_score)


if __name__ == '__main__':
    app.run(port=5001)
