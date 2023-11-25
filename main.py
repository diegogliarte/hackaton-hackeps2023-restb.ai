import cv2
from dotenv import load_dotenv

from models import RestbAiModels
from processor import show_and_process_frames, classify_frames, upload_images, save_classified_frames
from restbai_api import get_best_photo


def get_best_photos(video_path, dev=False):
    models = [RestbAiModels.ROOM_TYPE, RestbAiModels.CONDITION]

    interval = 1.5

    frames, average_score = show_and_process_frames(interval, models, video_path, dev)
    rooms = classify_frames(frames)
    save_classified_frames(rooms)
    rooms = upload_images(rooms)
    best_photos = get_best_photo(rooms)

    result_best_photos = []
    for photo in best_photos:
        if photo:
            frame = photo["frame"]
            result_best_photos.append(frame)
            if dev:
                cv2.imshow("frame", frame)
                cv2.waitKey(0)

    return result_best_photos, average_score


if __name__ == "__main__":
    load_dotenv()

    video_path = "videos/1.mp4"
    get_best_photos(video_path, dev=True)
