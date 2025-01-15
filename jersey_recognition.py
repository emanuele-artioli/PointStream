import os
import csv
import argparse
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.image_utils import load_image

def perform_jersey_recognition(frames_segmented_folder, jersey_csv):
    if not os.path.isfile(jersey_csv):
        with open(jersey_csv, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(['image_path', 'jersey_number', 'player_name'])
        
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
            for subfolder_name in os.listdir(frames_segmented_folder):
                frame_count += 1
                if frame_count % 30 == 0:
                    subfolder_path = os.path.join(frames_segmented_folder, subfolder_name)
                    if os.path.isdir(subfolder_path):
                        image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        for image_file in image_files:
                            image_path = os.path.join(subfolder_path, image_file)
                            image = load_image(image_path)
                            if image is None:
                                continue

                            inputs = processor(text=prompt, images=[image], return_tensors="pt")
                            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                            
                            generated_ids = model.generate(**inputs, max_new_tokens=500)
                            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

                            values = []
                            for text in generated_texts:
                                digits = ','.join(text)
                                if digits:
                                    values.append(digits)

                            if values:
                                csv_writer.writerow([subfolder_name, image_file, values])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform jersey number recognition on segmented images.')
    parser.add_argument('--frames_segmented_folder', type=str, required=True, help='Path to the folder containing segmented images.')
    parser.add_argument('--jersey_csv', type=str, required=True, help='Path to the CSV file to save jersey number recognition data.')
    args = parser.parse_args()

    perform_jersey_recognition(args.frames_segmented_folder, args.jersey_csv)
