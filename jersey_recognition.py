import os
import csv
import argparse
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image
import easyocr
import random

def perform_jersey_recognition_idefics(segmented_objects_folder, jersey_csv):
    recognized_info = {}  # keyed by object_id, storing {"name": None or str, "number": None or str}
    if not os.path.isfile(jersey_csv):
        with open(jersey_csv, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['image_path', 'player_name', 'jersey_number'])
        
            DEVICE = "cpu"
            processor = AutoProcessor.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3")
            model = AutoModelForVision2Seq.from_pretrained("HuggingFaceM4/Idefics3-8B-Llama3", torch_dtype=torch.bfloat16).to(DEVICE)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Here is a picture of a soccer player. Reply with only the player name and jersey number, if you recognise them."},
                    ]
                },       
            ]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            frame_count = 0
            for subfolder_name in os.listdir(segmented_objects_folder):
                frame_count += 1
                if frame_count % 30 == 0:
                    subfolder_path = os.path.join(segmented_objects_folder, subfolder_name)
                    if os.path.isdir(subfolder_path):
                        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        # random.shuffle(image_files)
                        skip_n = 10
                        check_m = 5
                        low_thresh = 0.25
                        high_thresh = 0.5
                        have_low = {"name": None, "number": None}
                        frame_list = sorted(image_files)
                        idx = 0
                        while idx < len(frame_list):
                            image_file = frame_list[idx]
                            object_id, _ = os.path.splitext(image_file)
                            if (object_id in recognized_info and
                                recognized_info[object_id]["name"] and
                                recognized_info[object_id]["number"]):
                                continue

                            image_path = os.path.join(subfolder_path, image_file)
                            image = load_image(image_path)
                            if image is None:
                                continue

                            inputs = processor(text=prompt, images=[image], return_tensors="pt")
                            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                            
                            generated_ids = model.generate(**inputs, max_new_tokens=500)
                            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                            for text in generated_texts:
                                prob = 0.3  # example, replace with actual measure
                                if prob >= low_thresh:
                                    # tentatively store partial name/number
                                    partial_name = text if text and not text.isdigit() else None
                                    partial_number = text if text and text.isdigit() else None
                                    next_frames = frame_list[idx+1:idx+1+check_m]
                                    confirmed = False
                                    for nf in next_frames:
                                        # ...existing code to load next image, generate prob (nprob)...
                                        nprob = 0.6  # example
                                        if nprob >= high_thresh:
                                            # confirm partial info
                                            if partial_name: 
                                                recognized_info.setdefault(object_id, {"name": None, "number": None})["name"] = partial_name
                                            if partial_number: 
                                                recognized_info.setdefault(object_id, {"name": None, "number": None})["number"] = partial_number
                                            confirmed = True
                                            break
                                    if not confirmed:
                                        # revert to partial (keep them only if needed)
                                        if partial_name and not recognized_info.get(object_id, {}).get("name"):
                                            recognized_info.setdefault(object_id, {})["name"] = partial_name
                                        if partial_number and not recognized_info.get(object_id, {}).get("number"):
                                            recognized_info.setdefault(object_id, {})["number"] = partial_number
                                    idx += check_m
                                    break
                                else:
                                    idx += skip_n
                                    break

                                if text.isdigit():
                                    recognized_info.setdefault(object_id, {"name": None, "number": None})["number"] = text
                                else:
                                    recognized_info.setdefault(object_id, {"name": None, "number": None})["name"] = text

                                name = recognized_info[object_id]["name"]
                                number = recognized_info[object_id]["number"]
                                if name and number:
                                    csv_writer.writerow([os.path.join(subfolder_path, image_file), name, number])
                                    break  # move to next subfolder

def perform_jersey_recognition_easyocr(segmented_objects_folder, jersey_csv):
    recognized_info = {}
    with open(jersey_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['image_path', 'player_name', 'jersey_number'])
        reader = easyocr.Reader(['en'], model_storage_directory="/app", gpu=True)

        for subfolder_name in os.listdir(segmented_objects_folder):
            subfolder_path = os.path.join(segmented_objects_folder, subfolder_name)
            if os.path.isdir(subfolder_path):
                image_files = sorted(
                    (f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))),
                    key=lambda x: int(os.path.splitext(x)[0])
                )
                skip_n = 10
                check_m = 5
                low_thresh = 0.25
                high_thresh = 0.5
                frame_list = sorted(image_files)
                idx = 0
                while idx < len(frame_list):
                    image_file = frame_list[idx]
                    object_id, _ = os.path.splitext(image_file)                        
                    if (object_id in recognized_info and                            
                    recognized_info[object_id]["name"] and                            
                    recognized_info[object_id]["number"]):                            
                        continue                        
                    image_path = os.path.join(subfolder_path, image_file)
                    result = reader.readtext(image_path)                        
                    found_something = False
                    for (bbox, text, prob) in result:                            
                        if prob >= low_thresh:
                            partial_name = text if text and not text.isdigit() else None
                            partial_number = text if text and text.isdigit() else None
                            next_frames = frame_list[idx+1:idx+1+check_m]
                            confirmed = False
                            for nf in next_frames:
                                # ...existing code to read next frame, produce nprob...
                                nprob = 0.6  # example
                                if nprob >= high_thresh:
                                    if partial_name: 
                                        recognized_info.setdefault(object_id, {"name": None, "number": None})["name"] = partial_name
                                    if partial_number: 
                                        recognized_info.setdefault(object_id, {"name": None, "number": None})["number"] = partial_number
                                    confirmed = True
                                    break
                            if not confirmed:
                                if partial_name and not recognized_info.get(object_id, {}).get("name"):
                                    recognized_info.setdefault(object_id, {})["name"] = partial_name
                                if partial_number and not recognized_info.get(object_id, {}).get("number"):
                                    recognized_info.setdefault(object_id, {})["number"] = partial_number
                            idx += check_m
                            found_something = True
                            break
                    if not found_something:
                        idx += skip_n

                    name = recognized_info.get(object_id, {}).get("name")
                    number = recognized_info.get(object_id, {}).get("number")
                    if name and number:
                        csv_writer.writerow([os.path.join(subfolder_path, image_file), name, number])
                        break

# TODO: save the models in the Docker so they are not downloaded at each run
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform jersey number recognition on segmented images.')
    parser.add_argument('--segmented_objects_folder', type=str, required=True, help='Path to the folder containing segmented object images.')
    parser.add_argument('--jersey_csv', type=str, required=True, help='Path to the CSV file to save jersey recognition data.')
    parser.add_argument('--method', type=str, required=True, choices=['idefics', 'easyocr'], help='Method to use for jersey recognition.')
    args = parser.parse_args()

    if args.method == 'idefics':
        perform_jersey_recognition_idefics(args.segmented_objects_folder, args.jersey_csv)
    elif args.method == 'easyocr':
        perform_jersey_recognition_easyocr(args.segmented_objects_folder, args.jersey_csv)
