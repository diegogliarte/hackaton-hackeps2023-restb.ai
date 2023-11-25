import json
import os
from enum import Enum
from typing import List

import numpy as np
import requests

from image import image_to_base64

URL = 'https://api-us.restb.ai/vision/v2/multipredict'


def call_service(models: List[Enum], image: np.ndarray) -> dict:
    model_ids = ','.join(model.value for model in models)

    payload = {
        "client_key": os.environ.get("RESTB_CLIENT_KEY"),
        'model_id': model_ids,
        "image_url": "",
    }
    data = {
        "image_base64": image_to_base64(image)
    }

    response = requests.get(URL, params=payload, data=data)
    if response.status_code != 200:
        raise Exception(f'Error calling Restb.ai API: {response.status_code} - {response.text}')
    response_json = response.json()
    return response_json


def get_best_photo(rooms):
    url = 'https://property.restb.ai/v1/multianalyze'
    payload = {
        'client_key': os.environ.get("RESTB_CLIENT_KEY"),
    }

    best_photos = []
    for room in rooms:
        links = [frame["link"] for frame in room]
        request_body = {
            "image_urls": links,
            "solutions": {"roomtype": 1.0}
        }

        response = requests.post(url, params=payload, json=request_body)

        json_response = response.json()
        best_photo = json_response["response"]["solutions"]["roomtype"]["summary"]["best_photo"]


        best_photo_label = room[0]["label"]
        best_photo_number = best_photo.get(best_photo_label, None)
        if not best_photo_number:
            for name, number in best_photo.items():
                if number:
                    best_photo_label = name
                    best_photo_number = number
                    break

        if best_photo_number is not None:
            best_photo_index = get_photo_index(json_response, best_photo_label, best_photo_number)
            if best_photo_index:
                best_frame = room[best_photo_index]
                best_photos.append(best_frame)
            else:
                best_photos.append(room[len(room) // 2])
        else:
            best_photos.append(room[len(room) // 2])

    return best_photos


def get_photo_index(json_response, label, occurrence):
    results = json_response["response"]["solutions"]["roomtype"]["results"]
    label_count = 0

    for i, result in enumerate(results):
        top_prediction = result["values"]["top_prediction"]
        if top_prediction["label"] == label:
            if label_count == occurrence:
                return i
            label_count += 1

    return None
