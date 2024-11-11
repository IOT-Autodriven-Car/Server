from ultralytics import YOLO
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
best_pt_path = os.path.join(current_directory, 'best.pt')

model = YOLO(best_pt_path)