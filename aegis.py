import cv2
import os
import numpy as np
import easyocr
from concurrent.futures import ThreadPoolExecutor
import re
import time
import warnings
from collections import defaultdict
import pyfiglet

title = pyfiglet.figlet_format("Aegis")

print(title)
print("         by Hyrulien\n")

# Suppress specific PyTorch warning
warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk of memory")

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Updated region with additional width and height
regions = [(867, 204, 1782, 547)]

def extract_damage_numbers(frame, regions):
    damage_numbers = defaultdict(lambda: {'number': None, 'confidence': 0})
    
    for roi in regions:
        x1, y1, x2, y2 = roi
        # Crop the region of interest
        cropped_frame = frame[y1:y2, x1:x2]

        # Convert frame to grayscale
        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
        
        # Use EasyOCR to read text from the grayscale image
        results = reader.readtext(gray)
        
        for (bbox, text, prob) in results:
            text = text.strip()
            # Filter out non-damage numbers
            if text.isdigit() and len(text) >= 4:  # Adjust minimum length as per observations
                number = text.lstrip('0')  # Remove leading zeros
                if number:
                    number = int(number)
                    # Update confidence score if higher than previous detections
                    if prob > damage_numbers[number]['confidence']:
                        damage_numbers[number] = {'number': number, 'confidence': prob}

    # Write filtered numbers to confidence.txt
    with open('confidence.txt', 'a') as f:
        for number, data in damage_numbers.items():
            if data['confidence'] >= 0.97:
                f.write(f"Detected text: {data['number']}, Confidence score: {data['confidence']}\n")
        
    return damage_numbers

def filter_damage_numbers(damage_numbers):
    filtered_numbers = set()
    critical_numbers = set()
    
    for number, data in damage_numbers.items():
        if data['confidence'] >= 0.97 and len(str(number)) >= 6:  # Filter out numbers with less than 6 digits
            if number > 750000:
                critical_numbers.add(f"{number} - crit")
            else:
                filtered_numbers.add(number)
    
    return filtered_numbers, critical_numbers

def process_frame(frame, regions):
    return extract_damage_numbers(frame, regions)

def process_video(video_path, output_path, regions):
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # # Set up video writer for debug video
    # debug_video_path = output_path.replace('.txt', '_debug.mp4')
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter(debug_video_path, fourcc, fps, (frame_width, frame_height))
    
    all_damage_numbers = set()
    recent_damage_numbers = set()
    frame_count = 0
    recent_frames_window = 3  # Adjust this value based on the video frame rate and typical duration of damage numbers
    
    # Start timing the processing
    start_time = time.time()
    
    # Create a ThreadPoolExecutor to process frames in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw regions of interest on the frame
            for roi in regions:
                x1, y1, x2, y2 = roi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # # Write the frame to the debug video
            # out.write(frame)
            
            # Submit frame for processing
            futures.append(executor.submit(process_frame, frame, regions))
            
            frame_count += 1
            # Calculate and print progress and ETA
            elapsed_time = time.time() - start_time
            progress = (frame_count / total_frames) * 100
            eta = (elapsed_time / frame_count) * (total_frames - frame_count)
            print(f"Processing video: {progress:.2f}% complete. ETA: {eta:.2f} seconds", end='\r')
            
            # Clear recent_damage_numbers periodically to allow new detections
            if frame_count % recent_frames_window == 0:
                recent_damage_numbers.clear()
        
        for future in futures:
            damage_numbers = future.result()
            for number, data in damage_numbers.items():
                if data['number'] not in recent_damage_numbers and data['confidence'] >= 0.97:
                    all_damage_numbers.add(data['number'])
                    recent_damage_numbers.add(data['number'])
    
    cap.release()
    #out.release()
    
    # Read filtered damage numbers from confidence.txt
    filtered_numbers = set()
    critical_numbers = set()
    with open('confidence.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = re.search(r'Detected text: (\d+), Confidence score: (\d+\.\d+)', line)
            if match:
                number = int(match.group(1))
                confidence = float(match.group(2))
                if confidence >= 0.97 and len(str(number)) >= 6:
                    if number in all_damage_numbers and confidence > 0.97:
                        if number > 750000:
                            critical_numbers.add(f"{number} - crit")
                        else:
                            filtered_numbers.add(number)
    
    # Write filtered damage numbers to output.txt
    with open(output_path, 'w') as f:
        for number in sorted(filtered_numbers):
            f.write(f"{number}\n")
        for number in sorted(critical_numbers):
            f.write(f"{number}\n")
    
    # Start timing the text file creation
    start_time_text_file = time.time()
    
    # Calculate and print the time taken to create the text file
    elapsed_time_text_file = time.time() - start_time_text_file
    print(f"\nText file created in {elapsed_time_text_file:.2f} seconds")
    
    # Calculate and print the total time taken
    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    # print(f"Debug video saved to {debug_video_path}")

def main():
    # Define input and output directories
    input_dir = './TestingVideos'
    output_dir = './results'
    # template_dir = './DamageTemplates'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Process each video in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.mp4'):
            video_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
            print(f"Starting processing {filename}")
            print("Warning: This process can take up to 8 minutes.")
            process_video(video_path, output_path, regions)
            print(f"\nProcessed {filename}, results saved to {output_path}")

if __name__ == '__main__':
    main()
