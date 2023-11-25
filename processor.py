import os
import time
import uuid
from multiprocessing import Process, Queue

import cv2
import requests

from image import image_to_base64
from restbai_api import call_service


def frame_processor(input_queue, output_queue, models):
    while True:
        frame = input_queue.get()
        if frame is None:
            output_queue.put(None)
            break

        response = call_service(models=models, image=frame)
        if response:
            output_queue.put(response)


def send_request(frame, input_queue, interval, prev_time):
    current_time = time.time()
    if current_time - prev_time > interval:
        prev_time = current_time
        if input_queue.qsize() < 10:
            input_queue.put(frame.copy())
    return prev_time


def show_and_process_frames(interval, models, video_path, dev):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) * 3
    input_queue = Queue(maxsize=10)
    output_queue = Queue()
    processor_process = Process(target=frame_processor, args=(input_queue, output_queue, models))
    processor_process.start()
    prev_time = 0
    last_label = None
    last_confidence = None
    last_score = None
    total_score = 0
    count_score = 0
    frames = []
    i = 0
    save_frame_every = 10
    while cap.isOpened():
        ret, frame = cap.read()
        i += 1
        if not ret:
            break
        resized_frame = cv2.resize(frame, (800, 600))

        prev_time = send_request(resized_frame, input_queue, interval, prev_time)
        if not output_queue.empty():
            last_confidence, last_label, score = get_response(last_confidence, last_label, models, output_queue)
            if score is not None:
                total_score += score
                count_score += 1
                last_score = score
        if i % save_frame_every == 0:
            frames.append({
                "frame": frame.copy(),
                "label": last_label,
                "confidence": last_confidence,
                "score": last_score
            })

        if dev:
            average_score = total_score / count_score if count_score > 0 else 0
            show_text(frame, last_label, last_confidence, average_score)
            resized_frame = cv2.resize(frame, (1280, 720 ))

            cv2.imshow("Restb.AI Analysis", resized_frame)

            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    input_queue.put(None)
    processor_process.join()
    cv2.destroyAllWindows()
    cap.release()
    return frames, total_score / count_score if count_score > 0 else 0


def get_response(last_confidence, last_label, models, output_queue):
    response = output_queue.get()
    top_prediction = response["response"]["solutions"][models[0].value]["top_prediction"]
    score = response["response"]["solutions"][models[1].value]["score"]
    if top_prediction is not None:
        last_label = top_prediction['label']
        last_confidence = top_prediction['confidence']
    return last_confidence, last_label, score


def show_text(frame, last_label, last_confidence, average_score):
    logo_path = "logo.png"
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

    scale_percent = 10
    width = int(logo.shape[1] * scale_percent / 100)
    height = int(logo.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_logo = cv2.resize(logo, dim, interpolation=cv2.INTER_AREA)

    frame_height, frame_width = frame.shape[:2]

    logo_x, logo_y = 10, 10
    rectangle_margin = 50

    rectangle_start_point = (frame_width - 200 - rectangle_margin, 10)
    rectangle_end_point = (frame_width - 10, logo_y + height + rectangle_margin)

    logo_x = rectangle_start_point[0] - 5
    logo_y = rectangle_start_point[1] - 10

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (0, 0, 0)
    last_label = last_label.replace('_', ' ').title() if last_label is not None else None
    last_confidence = last_confidence * 100 if last_confidence is not None else None
    value_text = f'\n{last_label}\n{last_confidence:.2f}%\nHouse Score: R{average_score:.2f}' if last_label is not None else '\nProcessing...'
    value_text_lines = value_text.split('\n')

    cv2.rectangle(frame, rectangle_start_point, rectangle_end_point, (255, 255, 255), cv2.FILLED)
    cv2.rectangle(frame, rectangle_start_point, rectangle_end_point, (0, 165, 255), 2)

    for i in range(0, 3):
        frame[logo_y:logo_y + height, logo_x:logo_x + width, i] = resized_logo[:, :, i] * (
                resized_logo[:, :, 3] / 255.0) + frame[logo_y:logo_y + height, logo_x:logo_x + width, i] * (
                                                                          1.0 - resized_logo[:, :, 3] / 255.0)

    for i, line in enumerate(value_text_lines):
        text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
        text_x = rectangle_start_point[0] + 10
        text_y = rectangle_start_point[1] + text_size[1] + 10 + (i * (text_size[1] + 10))
        cv2.putText(frame, line, (text_x, text_y), font, font_scale, text_color, font_thickness)

    return frame


def classify_frames(frames):
    room_groups = []
    current_group = []
    previous_label = None
    transition_threshold = 5

    for frame_info in frames:
        current_label = frame_info['label']

        # Skip frames where the label is None at the beginning
        if current_label is None and previous_label is None:
            continue

        # Start a new group if the current label is different from the previous
        if current_label != previous_label and previous_label is not None:
            if len(current_group) >= transition_threshold:
                room_groups.append(current_group)
            current_group = [frame_info]
        else:
            # Add to current group
            current_group.append(frame_info)

        previous_label = current_label

    # Add the last group if it meets the threshold
    if len(current_group) >= transition_threshold:
        room_groups.append(current_group)

    # Assuming merge_small_transition_groups is defined elsewhere
    room_groups = merge_small_transition_groups(room_groups, transition_threshold)

    return room_groups

def merge_small_transition_groups(groups, threshold):
    optimized_groups = []
    i = 0
    while i < len(groups):
        group = groups[i]
        next_group = groups[i + 1] if i + 1 < len(groups) else None

        if next_group and len(group) < threshold:
            next_label = next_group[0]['label']
            if all(frame['label'] == next_label for frame in group):
                next_group = group + next_group
                groups[i + 1] = next_group
                i += 1
            else:
                optimized_groups.append(group)
        else:
            optimized_groups.append(group)

        i += 1

    return optimized_groups


def save_classified_frames(classified_rooms):
    room_counts = {}
    folder = str(uuid.uuid4())
    for group in classified_rooms:
        if not group:
            continue

        room_label = group[0]['label']
        room_counts[room_label] = room_counts.get(room_label, 0) + 1
        folder_name = f"classified_frames/{folder}/{room_label}_{room_counts[room_label]:02d}"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        for i, frame_info in enumerate(group):
            frame = frame_info['frame']
            file_name = os.path.join(folder_name, f"frame_{i:03d}.jpg")
            cv2.imwrite(file_name, frame)


def upload_to_server(images):
    files = []

    for image_path in images:
        image_b64 = image_to_base64(image_path)
        file = {'image': image_b64,
                "filename": f"{str(uuid.uuid4())}.jpg"}
        files.append(file)

    server_url = "http://nicodeco.love:5000/upload"
    response = requests.post(server_url, json={"files": files})

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to upload image: {response.text}")
        return None


def upload_images(rooms):
    rooms_with_links = rooms.copy()
    for room_index, room in enumerate(rooms):
        room_name = room[0]['label']
        print(f"Uploading images for {room_name}...")

        images = [image["frame"] for image in room]

        links = upload_to_server(images)
        for idx, link in enumerate(links):
            rooms_with_links[room_index][idx]["link"] = link
    return rooms_with_links
