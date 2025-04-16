import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.signal import find_peaks
from skimage import measure, segmentation, color, morphology
from sklearn.cluster import DBSCAN
import glob
import re

class PoppyAnalyzer:
    def __init__(self, image_dir, output_dir, time_interval=10):
        """
        Initialize the PoppyAnalyzer class.
        
        Args:
            image_dir (str): Directory containing time-lapse images
            output_dir (str): Directory for saving results
            time_interval (int): Minutes between each image
        """
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.time_interval = time_interval
        self.images = []
        self.timestamps = []
        self.flower_tracks = {}
        self.flower_data = pd.DataFrame()
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def load_images(self, start_time=None):
        """
        Load images from the image directory and sort them chronologically.
        
        Args:
            start_time (datetime, optional): Starting time for the time-lapse.
                If None, will try to extract from filenames.
        """
        print("Loading images...")
        image_files = sorted(glob.glob(os.path.join(self.image_dir, "*.jpg")) + 
                            glob.glob(os.path.join(self.image_dir, "*.jpeg")) + 
                            glob.glob(os.path.join(self.image_dir, "*.png")))
        
        if not image_files:
            raise ValueError(f"No image files found in {self.image_dir}")
        
        # Try to extract timestamps from filenames if available
        if start_time is None:
            # Check if filenames contain timestamps
            sample_file = os.path.basename(image_files[0])
            timestamp_pattern = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})', sample_file)
            
            if timestamp_pattern:
                print("Extracting timestamps from filenames...")
                current_time = None
            else:
                print("Using sequence number for timestamps...")
                # Use the first image timestamp as starting point
                current_time = datetime.now().replace(hour=6, minute=0, second=0)
                print(f"Setting start time to {current_time}")
        else:
            current_time = start_time
        
        for i, img_file in enumerate(image_files):
            try:
                img = cv2.imread(img_file)
                if img is None:
                    print(f"Warning: Could not read {img_file}, skipping.")
                    continue
                
                self.images.append(img)
                
                # Get timestamp from filename or calculate based on interval
                if current_time is None:
                    filename = os.path.basename(img_file)
                    timestamp_match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})', filename)
                    if timestamp_match:
                        year, month, day, hour, minute = map(int, timestamp_match.groups())
                        timestamp = datetime(year, month, day, hour, minute)
                    else:
                        timestamp = datetime.now() + timedelta(minutes=i*self.time_interval)
                else:
                    timestamp = current_time + timedelta(minutes=i*self.time_interval)
                
                self.timestamps.append(timestamp)
                
                if i % 10 == 0:
                    print(f"Loaded {i+1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"Error loading {img_file}: {str(e)}")
        
        print(f"Successfully loaded {len(self.images)} images.")
    
    def preprocess_images(self):
        """Preprocess images for analysis (resize, normalize, etc.)"""
        print("Preprocessing images...")
        processed_images = []
        
        for img in self.images:
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize for consistency if needed
            # height, width = img_rgb.shape[:2]
            # if width > 1000:
            #     scale_factor = 1000 / width
            #     img_rgb = cv2.resize(img_rgb, None, fx=scale_factor, fy=scale_factor)
            
            # Convert to HSV for better color segmentation
            img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
            
            processed_images.append((img_rgb, img_hsv))
        
        self.processed_images = processed_images
        return processed_images
    
    def detect_flowers(self):
        """Detect orange poppy flowers in the images"""
        print("Detecting flowers in each frame...")
        all_flowers = []
        
        for i, (img_rgb, img_hsv) in enumerate(self.processed_images):
            # Create mask for orange flowers
            # California poppy color range in HSV
            lower_orange = np.array([10, 100, 100])
            upper_orange = np.array([30, 255, 255])
            mask = cv2.inRange(img_hsv, lower_orange, upper_orange)
            
            # Apply morphological operations to clean the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask = cv2.erode(mask, kernel, iterations=1)
            
            # Find connected components in the mask
            labels = measure.label(mask)
            props = measure.regionprops(labels)
            
            # Filter regions by area to find flowers
            flowers = []
            min_area = 100  # Minimum area in pixels
            max_area = 10000  # Maximum area in pixels
            
            for prop in props:
                if min_area < prop.area < max_area:
                    y, x = prop.centroid
                    bbox = prop.bbox  # (min_row, min_col, max_row, max_col)
                    area = prop.area
                    
                    # Calculate average color in the region
                    flower_mask = labels == prop.label
                    flower_rgb = img_rgb.copy()
                    flower_rgb[~np.dstack([flower_mask]*3)] = 0
                    non_zero = flower_rgb[flower_mask]
                    avg_color = np.mean(non_zero, axis=0) if len(non_zero) > 0 else [0, 0, 0]
                    
                    # Calculate flower "openness" based on area and shape
                    perimeter = prop.perimeter
                    eccentricity = prop.eccentricity
                    solidity = prop.solidity  # Area / ConvexHullArea
                    
                    # Store flower features
                    flower = {
                        'frame': i,
                        'id': -1,  # Will be assigned during tracking
                        'centroid': (x, y),
                        'bbox': bbox,
                        'area': area,
                        'perimeter': perimeter,
                        'eccentricity': eccentricity,
                        'solidity': solidity,
                        'avg_color': avg_color,
                        'timestamp': self.timestamps[i]
                    }
                    flowers.append(flower)
            
            all_flowers.append(flowers)
            
            if i % 10 == 0:
                print(f"Processed {i+1}/{len(self.processed_images)} frames")
        
        self.all_detected_flowers = all_flowers
        return all_flowers
    
    def track_flowers(self):
        """Track individual flowers across frames"""
        print("Tracking flowers across frames...")
        flower_id = 0
        tracks = {}
        
        for frame_idx, flowers in enumerate(self.all_detected_flowers):
            if frame_idx == 0:
                # Initialize tracks with flowers from first frame
                for flower in flowers:
                    flower['id'] = flower_id
                    tracks[flower_id] = [flower]
                    flower_id += 1
            else:
                # Get previous frame flowers with IDs
                prev_flowers = self.all_detected_flowers[frame_idx - 1]
                
                # Extract centroids from previous flowers
                if prev_flowers:
                    prev_centroids = np.array([flower['centroid'] for flower in prev_flowers])
                    prev_ids = [flower['id'] for flower in prev_flowers]
                    
                    # Match flowers based on position
                    for flower in flowers:
                        if prev_centroids.size == 0:
                            # No flowers in previous frame to match with
                            flower['id'] = flower_id
                            tracks[flower_id] = [flower]
                            flower_id += 1
                            continue
                            
                        centroid = np.array(flower['centroid'])
                        # Calculate distances to all previous flowers
                        distances = np.sqrt(np.sum((prev_centroids - centroid)**2, axis=1))
                        
                        # Find the closest previous flower
                        min_dist_idx = np.argmin(distances)
                        min_dist = distances[min_dist_idx]
                        
                        # If the closest flower is within threshold distance, assign same ID
                        max_distance = 50  # Maximum allowed distance for the same flower
                        if min_dist < max_distance:
                            matched_id = prev_ids[min_dist_idx]
                            flower['id'] = matched_id
                            tracks[matched_id].append(flower)
                        else:
                            # This is a new flower
                            flower['id'] = flower_id
                            tracks[flower_id] = [flower]
                            flower_id += 1
                else:
                    # No flowers in previous frame
                    for flower in flowers:
                        flower['id'] = flower_id
                        tracks[flower_id] = [flower]
                        flower_id += 1
        
        # Filter tracks that don't have enough observations
        min_track_length = 5  # Minimum number of frames a flower should appear in
        filtered_tracks = {k: v for k, v in tracks.items() if len(v) >= min_track_length}
        
        print(f"Found {len(filtered_tracks)} consistent flower tracks.")
        self.flower_tracks = filtered_tracks
        return filtered_tracks
    
    def analyze_flower_states(self):
        """Analyze opening and closing states of each tracked flower"""
        print("Analyzing flower opening and closing states...")
        flower_data = []
        
        for flower_id, track in self.flower_tracks.items():
            # Extract timestamps and metrics for this flower
            timestamps = [point['timestamp'] for point in track]
            areas = [point['area'] for point in track]
            perimeters = [point['perimeter'] for point in track]
            solidities = [point['solidity'] for point in track]
            
            # Sort by timestamp to ensure chronological order
            sorted_indices = np.argsort(timestamps)
            timestamps = [timestamps[i] for i in sorted_indices]
            areas = [areas[i] for i in sorted_indices]
            perimeters = [perimeters[i] for i in sorted_indices]
            solidities = [solidities[i] for i in sorted_indices]
            
            # Normalize metrics
            if len(areas) > 1:
                areas_norm = (np.array(areas) - np.min(areas)) / (np.max(areas) - np.min(areas) + 1e-10)
                
                # Smooth the data to reduce noise
                window_size = min(5, len(areas))
                areas_smooth = np.convolve(areas_norm, np.ones(window_size)/window_size, mode='valid')
                
                # Find potential opening and closing events
                # We'll use the derivative of area to detect changes
                if len(areas_smooth) > 1:
                    derivatives = np.diff(areas_smooth)
                    
                    # Find where derivative crosses threshold (significant changes)
                    open_threshold = 0.05
                    close_threshold = -0.05
                    
                    # Potential opening starts when derivative becomes positive
                    opening_indices = [i for i, d in enumerate(derivatives) if d > open_threshold]
                    
                    # Potential closing starts when derivative becomes negative
                    closing_indices = [i for i, d in enumerate(derivatives) if d < close_threshold]
                    
                    # Group consecutive indices to find start/end of events
                    open_events = []
                    if opening_indices:
                        current_event = [opening_indices[0]]
                        for i in range(1, len(opening_indices)):
                            if opening_indices[i] - opening_indices[i-1] <= 2:  # Consecutive or nearly consecutive
                                current_event.append(opening_indices[i])
                            else:
                                if len(current_event) > 0:
                                    open_events.append((current_event[0], current_event[-1]))
                                current_event = [opening_indices[i]]
                        if len(current_event) > 0:
                            open_events.append((current_event[0], current_event[-1]))
                    
                    close_events = []
                    if closing_indices:
                        current_event = [closing_indices[0]]
                        for i in range(1, len(closing_indices)):
                            if closing_indices[i] - closing_indices[i-1] <= 2:  # Consecutive or nearly consecutive
                                current_event.append(closing_indices[i])
                            else:
                                if len(current_event) > 0:
                                    close_events.append((current_event[0], current_event[-1]))
                                current_event = [closing_indices[i]]
                        if len(current_event) > 0:
                            close_events.append((current_event[0], current_event[-1]))
                    
                    # Adjust indices to account for smoothing window
                    window_offset = window_size - 1
                    
                    # Find most significant open/close events
                    open_start = None
                    open_complete = None
                    close_start = None
                    close_complete = None
                    
                    if open_events:
                        # Find most significant opening event
                        most_sig_open = max(open_events, key=lambda x: areas_smooth[x[1]] - areas_smooth[x[0]])
                        open_start_idx = most_sig_open[0] + window_offset//2
                        open_complete_idx = most_sig_open[1] + window_offset//2
                        
                        if 0 <= open_start_idx < len(timestamps) and 0 <= open_complete_idx < len(timestamps):
                            open_start = timestamps[open_start_idx]
                            open_complete = timestamps[open_complete_idx]
                    
                    if close_events:
                        # Find most significant closing event
                        most_sig_close = max(close_events, key=lambda x: abs(areas_smooth[x[1]] - areas_smooth[x[0]]))
                        close_start_idx = most_sig_close[0] + window_offset//2
                        close_complete_idx = most_sig_close[1] + window_offset//2
                        
                        if 0 <= close_start_idx < len(timestamps) and 0 <= close_complete_idx < len(timestamps):
                            close_start = timestamps[close_start_idx]
                            close_complete = timestamps[close_complete_idx]
                    
                    # Store the results for this flower
                    flower_result = {
                        'flower_id': flower_id,
                        'first_detected': timestamps[0] if timestamps else None,
                        'last_detected': timestamps[-1] if timestamps else None,
                        'open_start': open_start,
                        'open_complete': open_complete,
                        'close_start': close_start,
                        'close_complete': close_complete,
                        'max_area': max(areas) if areas else 0,
                        'track_length': len(track)
                    }
                    flower_data.append(flower_result)
        
        # Convert to DataFrame
        self.flower_data = pd.DataFrame(flower_data)
        return self.flower_data
    
    def generate_visualizations(self):
        """Generate visualizations of flower tracking and states"""
        print("Generating visualizations...")
        
        # Create output directory for visualizations
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # 1. Create tracking visualization for a few frames
        sample_frames = min(20, len(self.images))
        track_vis_dir = os.path.join(vis_dir, "tracking")
        os.makedirs(track_vis_dir, exist_ok=True)
        
        # Generate color map for IDs
        ids = list(self.flower_tracks.keys())
        colors = {}
        for i, flower_id in enumerate(ids):
            # Generate distinctive colors
            hue = (i * 30) % 180  # Spread colors around hue space
            colors[flower_id] = tuple(int(c) for c in colorsys.hsv_to_rgb(hue/180, 0.9, 0.9))
        
        for frame_idx in range(0, len(self.images), len(self.images)//sample_frames):
            if frame_idx >= len(self.images):
                continue
                
            img = self.images[frame_idx].copy()
            
            # Draw all flowers in this frame
            for flower_list in self.all_detected_flowers[frame_idx]:
                flower_id = flower_list['id']
                if flower_id in self.flower_tracks:  # Only draw tracked flowers
                    x, y = map(int, flower_list['centroid'])
                    color = colors.get(flower_id, (255, 255, 255))
                    cv2.circle(img, (x, y), 10, color, -1)
                    cv2.putText(img, f"ID: {flower_id}", (x+10, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Save visualization
            cv2.imwrite(os.path.join(track_vis_dir, f"track_frame_{frame_idx:04d}.jpg"), img)
        
        # 2. Create state visualization for each tracked flower
        flower_vis_dir = os.path.join(vis_dir, "flowers")
        os.makedirs(flower_vis_dir, exist_ok=True)
        
        for flower_id, track in self.flower_tracks.items():
            # Get data for this flower
            flower_info = self.flower_data[self.flower_data['flower_id'] == flower_id]
            if flower_info.empty:
                continue
                
            # Extract timestamps and areas for plotting
            sorted_track = sorted(track, key=lambda x: x['timestamp'])
            timestamps = [point['timestamp'] for point in sorted_track]
            areas = [point['area'] for point in sorted_track]
            
            # Convert timestamps to numbers for plotting
            rel_times = [(t - timestamps[0]).total_seconds() / 60 for t in timestamps]
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            plt.plot(rel_times, areas, 'b-', linewidth=2)
            plt.title(f"Flower ID: {flower_id} - Area Over Time")
            plt.xlabel("Time (minutes)")
            plt.ylabel("Area (pixels)")
            
            # Mark the state transitions if available
            info = flower_info.iloc[0]
            
            # Mark open/close events
            events = []
            if info['open_start'] is not None:
                open_start_time = (info['open_start'] - timestamps[0]).total_seconds() / 60
                plt.axvline(x=open_start_time, color='g', linestyle='--', label='Open Start')
                events.append(('Open Start', open_start_time))
                
            if info['open_complete'] is not None:
                open_complete_time = (info['open_complete'] - timestamps[0]).total_seconds() / 60
                plt.axvline(x=open_complete_time, color='g', linestyle='-', label='Open Complete')
                events.append(('Open Complete', open_complete_time))
                
            if info['close_start'] is not None:
                close_start_time = (info['close_start'] - timestamps[0]).total_seconds() / 60
                plt.axvline(x=close_start_time, color='r', linestyle='--', label='Close Start')
                events.append(('Close Start', close_start_time))
                
            if info['close_complete'] is not None:
                close_complete_time = (info['close_complete'] - timestamps[0]).total_seconds() / 60
                plt.axvline(x=close_complete_time, color='r', linestyle='-', label='Close Complete')
                events.append(('Close Complete', close_complete_time))
            
            if events:
                plt.legend()
            
            plt.grid(True)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(flower_vis_dir, f"flower_{flower_id}_states.png"))
            plt.close()
        
        print(f"Visualizations saved to {vis_dir}")
    
    def save_results(self):
        """Save the analysis results to CSV"""
        if not self.flower_data.empty:
            result_file = os.path.join(self.output_dir, "flower_cycle_data.csv")
            self.flower_data.to_csv(result_file, index=False)
            print(f"Results saved to {result_file}")
            
            # Also save a summary file with readable timestamps
            summary_data = self.flower_data.copy()
            for col in ['first_detected', 'last_detected', 'open_start', 'open_complete', 
                        'close_start', 'close_complete']:
                summary_data[col] = summary_data[col].apply(
                    lambda x: x.strftime('%Y-%m-%d %H:%M') if pd.notnull(x) else 'N/A'
                )
            
            summary_file = os.path.join(self.output_dir, "flower_cycle_summary.csv")
            summary_data.to_csv(summary_file, index=False)
            print(f"Summary saved to {summary_file}")
        else:
            print("No flower data available to save.")
    
    def run_analysis(self, start_time=None):
        """Run the complete analysis pipeline"""
        self.load_images(start_time=start_time)
        self.preprocess_images()
        self.detect_flowers()
        self.track_flowers()
        self.analyze_flower_states()
        self.generate_visualizations()
        self.save_results()
        print("Analysis complete!")


# Import for visualization coloring
import colorsys

# Helper function to extract EXIF datetime from images if available
def get_image_datetime(image_path):
    """Try to extract datetime from image EXIF data"""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'DateTimeOriginal':
                    # Format: 'YYYY:MM:DD HH:MM:SS'
                    date_parts = value.split(' ')
                    if len(date_parts) == 2:
                        date = date_parts[0].replace(':', '-')
                        time = date_parts[1]
                        return datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M:%S")
        
        return None
    except Exception as e:
        print(f"Error getting EXIF datetime: {str(e)}")
        return None


def main():
    """Main function to run the analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze California Poppy time-lapse images')
    parser.add_argument('--input', '-i', required=True, help='Directory containing time-lapse images')
    parser.add_argument('--output', '-o', default='./poppy_results', help='Output directory for results')
    parser.add_argument('--interval', '-t', type=int, default=10, help='Time interval between images in minutes')
    parser.add_argument('--start-time', '-s', help='Start time in format YYYY-MM-DD HH:MM')
    
    args = parser.parse_args()
    
    # Parse start time if provided
    start_time = None
    if args.start_time:
        try:
            start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M")
        except ValueError:
            print("Invalid start time format. Please use YYYY-MM-DD HH:MM")
            return
    
    # Create and run the analyzer
    analyzer = PoppyAnalyzer(args.input, args.output, args.interval)
    analyzer.run_analysis(start_time=start_time)


if __name__ == "__main__":
    main()
