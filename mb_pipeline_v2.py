import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import re
import os
import sys

"""
MegaBouts Multi-Track Analysis Tool
This script processes multi-track fish tracking data from a CSV file, performs preprocessing, segmentation, and classification of fish movement bouts, and outputs the results in a simplified format.
Functions:
    - get_input_file_path(): Prompts the user to input the path to a CSV file and validates it.
    - get_track_selection(available_tracks): Allows the user to select tracks for analysis from the available tracks in the data.
    - get_certainty_threshold(): Prompts the user to input a certainty threshold for bout classification.
    - get_fps(): Prompts the user to input the FPS (frames per second) of the video data.
    - get_mm_per_unit(): Prompts the user to input the millimeters per unit of the data.
    - get_tail_points(): Prompts the user to input the number of tail points in the data.
    - process_track(df, track_num, fps, mm_per_unit, tail_points, libraries): Processes a single track and performs preprocessing, segmentation, and classification.
    - extract_simplified_results(bouts_df, track_num, fps, category_names, certainty_threshold): Extracts simplified results from the bouts dataframe.
    - main(): Main function that orchestrates the entire analysis process.
Usage:
    Run the script and follow the prompts to input the required parameters and analyze the data.
Dependencies:
    - numpy
    - pandas
    - matplotlib
    - megabouts (custom library)
    - torch
    - Other required libraries as specified in the script.
Notes:
    - Ensure all dependencies are installed before running the script.
    - The CSV file must contain a 'track' column and other required columns as specified in the script.
    - The script supports multi-track analysis and outputs results in a simplified format.
"""

def get_input_file_path():
    while True:
        file_path = input("Enter the relative path to your CSV file: ").strip()
        if not file_path:
            print("Error: File path cannot be empty.")
            continue
        
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' does not exist.")
            continue
            
        if not file_path.endswith('.csv'):
            print("Error: File must be a CSV file.")
            continue
            
        return file_path

def get_track_selection(available_tracks):
    print("\nAvailable tracks in the data:")
    track_nums = []
    
    # Extract numeric part from track names (assuming format "track_X")
    for track in available_tracks:
        try:
            if track.startswith("track_"):
                track_num = int(track.split("_")[1])
                track_nums.append(track_num)
        except (ValueError, IndexError):
            continue
    
    track_nums.sort()
    
    if not track_nums:
        print("No valid tracks found. Please check your data format.")
        sys.exit(1)
    
    print(f"Track numbers: {', '.join(str(num) for num in track_nums)}")
    
    while True:
        track_input = input(
            "\nEnter track number(s) to analyze:\n"
            "- Single track: e.g. '5'\n"
            "- Range of tracks: e.g. '1-4'\n"
            "- List of tracks: e.g. '1,3,5,7'\n"
            "Your selection: "
        ).strip()
        
        if not track_input:
            print("Error: Track selection cannot be empty.")
            continue
        
        selected_tracks = []
        
        # Check if input is a range (e.g., "1-5")
        if "-" in track_input and track_input.count("-") == 1 and "," not in track_input:
            try:
                start, end = map(int, track_input.split("-"))
                if start <= end and start > 0:
                    selected_tracks = list(range(start, end + 1))
                else:
                    print("Error: Invalid range. Start must be less than or equal to end, and both must be positive.")
                    continue
            except ValueError:
                print("Error: Range must contain valid integers.")
                continue
        
        # Check if input is a comma-separated list
        elif "," in track_input:
            try:
                nums = [int(num.strip()) for num in track_input.split(",")]
                if all(num > 0 for num in nums):
                    selected_tracks = nums
                else:
                    print("Error: All track numbers must be positive integers.")
                    continue
            except ValueError:
                print("Error: List must contain valid integers.")
                continue
        
        # Check if input is a single number
        else:
            try:
                num = int(track_input)
                if num > 0:
                    selected_tracks = [num]
                else:
                    print("Error: Track number must be a positive integer.")
                    continue
            except ValueError:
                print("Error: Please enter a valid integer.")
                continue
        
        # Validate all selected tracks exist in the data
        valid_tracks = []
        invalid_tracks = []
        
        for track_num in selected_tracks:
            if track_num in track_nums:
                valid_tracks.append(track_num)
            else:
                invalid_tracks.append(track_num)
        
        if invalid_tracks:
            print(f"Warning: The following tracks do not exist in the data: {', '.join(str(t) for t in invalid_tracks)}")
            
        if not valid_tracks:
            print("Error: None of the selected tracks exist in the data. Please try again.")
            continue
            
        return valid_tracks
    
def get_certainty_threshold():
    while True:
        threshold_input = input("Enter a certainty threshold (default: 0.5): ").strip()
        if not threshold_input:
            return 0.5
        
        try:
            threshold = float(threshold_input)
            if threshold < 0:
                print("Error: Certainty threshold must be greater than 0.")
                continue
            return threshold
        except ValueError:
            print("Error: Please enter a decimal number.")

def get_fps():
    while True:
        fps_input = input("Enter the FPS of the video data (default: 160): ").strip()
        if not fps_input:
            return 160
        
        try:
            fps = float(fps_input)
            if fps <= 0:
                print("Error: FPS must be greater than 0.")
                continue
            return fps
        except ValueError:
            print("Error: Please enter a valid number.")

def get_mm_per_unit():
    while True:
        mm_input = input("Enter the mm per unit of the data (default: 0.045): ").strip()
        if not mm_input:
            return 0.045
        
        try:
            mm_per_unit = float(mm_input)
            if mm_per_unit <= 0:
                print("Error: mm per unit must be greater than 0.")
                continue
            return mm_per_unit
        except ValueError:
            print("Error: Please enter a valid number.")

def get_tail_points():
    while True:
        tail_input = input("Enter the number of tail points (integer > 0): ").strip()
        try:
            tail_points = int(tail_input)
            if tail_points <= 0:
                print("Error: Number of tail points must be greater than 0.")
                continue
            return tail_points
        except ValueError:
            print("Error: Please enter a valid integer.")

def process_track(df, track_num, fps, mm_per_unit, tail_points, libraries):
    """Process a single track and return analysis results"""
    
    track_name = f"track_{track_num}"
    fish_data = df[df['track'] == track_name]
    
    if fish_data.empty:
        return None, f"No data found for track '{track_name}'."
    
    print(f"\n[Track {track_num}] Processing with {len(fish_data)} frames...")
    
    # Extract required libraries from the imported dictionary
    TrackingConfig = libraries['TrackingConfig']
    FullTrackingData = libraries['FullTrackingData']
    TailPreprocessingConfig = libraries['TailPreprocessingConfig']
    TailPreprocessing = libraries['TailPreprocessing']
    TrajPreprocessingConfig = libraries['TrajPreprocessingConfig']
    TrajPreprocessing = libraries['TrajPreprocessing']
    TailSegmentationConfig = libraries['TailSegmentationConfig']
    TrajSegmentationConfig = libraries['TrajSegmentationConfig']
    Segmentation = libraries['Segmentation']
    BoutClassifier = libraries['BoutClassifier']
    TailBouts = libraries['TailBouts']
    bouts_category_name_short = libraries['bouts_category_name_short']
    
    # Set up tracking configuration
    tracking_cfg = TrackingConfig(fps=fps, tracking="full_tracking")

    # Construct tail column names based on number of tail points
    tail_x_cols = [f"tail{i}.x" for i in range(1, tail_points + 1)]
    tail_y_cols = [f"tail{i}.y" for i in range(1, tail_points + 1)]
    
    # Check if required columns exist
    required_cols = ["leye.x", "leye.y", "reye.x", "reye.y"] + tail_x_cols + tail_y_cols
    missing_cols = [col for col in required_cols if col not in fish_data.columns]
    
    if missing_cols:
        return None, f"The following required columns are missing: {', '.join(missing_cols)}"

    # Apply threshold to scores
    thresh_score = 0.0
    print(f"[Track {track_num}] Filtering low-confidence keypoints...")
    
    for kps in ["leye", "reye"] + [f"tail{i}" for i in range(1, tail_points + 1)]:
        if f"{kps}.score" in fish_data.columns and "instance.score" in fish_data.columns:
            fish_data.loc[fish_data["instance.score"] < thresh_score, f"{kps}.x"] = np.nan
            fish_data.loc[fish_data["instance.score"] < thresh_score, f"{kps}.y"] = np.nan
            fish_data.loc[fish_data[f"{kps}.score"] < thresh_score, f"{kps}.x"] = np.nan
            fish_data.loc[fish_data[f"{kps}.score"] < thresh_score, f"{kps}.y"] = np.nan

    # Calculate head position
    head_x = (fish_data["leye.x"].values + fish_data["reye.x"].values) / 2
    head_y = (fish_data["leye.y"].values + fish_data["reye.y"].values) / 2

    # Extract tail data
    tail_x = fish_data[tail_x_cols].values
    tail_y = fish_data[tail_y_cols].values

    # Convert to mm
    head_x = head_x * mm_per_unit
    head_y = head_y * mm_per_unit
    tail_x = tail_x * mm_per_unit
    tail_y = tail_y * mm_per_unit

    # Create tracking data object
    print(f"[Track {track_num}] Creating tracking data object...")
    tracking_data = FullTrackingData.from_keypoints(
        head_x=head_x, head_y=head_y, tail_x=tail_x, tail_y=tail_y
    )

    # Preprocessing
    print(f"[Track {track_num}] Preprocessing tail data...")
    tail_preprocessing_cfg = TailPreprocessingConfig(fps=tracking_cfg.fps)
    tail_df_input = tracking_data.tail_df
    tail = TailPreprocessing(tail_preprocessing_cfg).preprocess_tail_df(tail_df_input)

    print(f"[Track {track_num}] Preprocessing trajectory data...")
    traj_preprocessing_cfg = TrajPreprocessingConfig(fps=tracking_cfg.fps)
    traj_df_input = tracking_data.traj_df
    traj = TrajPreprocessing(traj_preprocessing_cfg).preprocess_traj_df(traj_df_input)

    # Segmentation
    print(f"[Track {track_num}] Segmenting tail data...")
    tail_segmentation_cfg = TailSegmentationConfig(fps=tracking_cfg.fps, threshold=20)
    segmentation_function = Segmentation.from_config(tail_segmentation_cfg)
    segments = segmentation_function.segment(tail.vigor)

    print(f"[Track {track_num}] Segmenting trajectory data...")
    traj_segmentation_cfg = TrajSegmentationConfig(fps=tracking_cfg.fps, peak_prominence=1)
    segmentation_function = Segmentation.from_config(traj_segmentation_cfg)
    segments = segmentation_function.segment(traj.vigor)

    # Extract arrays
    tail_array = segments.extract_tail_array(tail_angle=tail.angle_smooth)
    traj_array = segments.extract_traj_array(
        head_x=traj.x_smooth, head_y=traj.y_smooth, head_angle=traj.yaw_smooth
    )

    # Classification
    print(f"[Track {track_num}] Running bout classification...")
    classifier = BoutClassifier(tracking_cfg, tail_segmentation_cfg, exclude_CS=False)
    classif_results = classifier.run_classification(
        tail_array=tail_array, traj_array=traj_array
    )

    segments.set_HB1(classif_results["first_half_beat"])

    tail_array = segments.extract_tail_array(
        tail_angle=tail.angle_smooth, align_to_onset=False
    )

    traj_array = segments.extract_traj_array(
        head_x=traj.x_smooth,
        head_y=traj.y_smooth,
        head_angle=traj.yaw_smooth,
        align_to_onset=False,
    )

    # Format Output
    print(f"[Track {track_num}] Formatting results...")
    bouts = TailBouts(
        segments=segments,
        classif_results=classif_results,
        tail_array=tail_array,
        traj_array=traj_array,
    )
    
    return bouts, None

def extract_simplified_results(bouts_df, track_num, fps, category_names, certainty_threshold):
    """Extract simplified results from the bouts dataframe"""
    results_list = []
    
    for index, row in bouts_df.iterrows():
        onset_frames = row[('location', 'onset')]
        category_id = int(row[('label', 'category')])
        certainty = row[('label', 'proba')]
        
        if certainty > certainty_threshold:  # Only include high-confidence bouts
            results_list.append({
                'track': track_num,
                'onset_frames': onset_frames,
                'onset_seconds': onset_frames / fps,
                'category': category_names[category_id],
                'certainty': certainty
            })
    
    return results_list

def main():
    print("\n=== MegaBouts Multi-Track Analysis Tool ===\n")

    try:
        # Import required libraries
        print("Importing required libraries...")
        libraries = {}
        try:
            from megabouts.tracking_data import (
                TrackingConfig,
                FullTrackingData,
                HeadTrackingData,
                TailTrackingData,
                load_example_data,
            )
            libraries['TrackingConfig'] = TrackingConfig
            libraries['FullTrackingData'] = FullTrackingData
            libraries['HeadTrackingData'] = HeadTrackingData
            libraries['TailTrackingData'] = TailTrackingData
            libraries['load_example_data'] = load_example_data

            from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            import matplotlib.patches as patches
            from cycler import cycler

            from megabouts.config import TailPreprocessingConfig, TailSegmentationConfig
            from megabouts.preprocessing import TailPreprocessing
            libraries['TailPreprocessingConfig'] = TailPreprocessingConfig
            libraries['TailSegmentationConfig'] = TailSegmentationConfig
            libraries['TailPreprocessing'] = TailPreprocessing

            from megabouts.config import TrajPreprocessingConfig, TrajSegmentationConfig
            from megabouts.preprocessing import TrajPreprocessing
            libraries['TrajPreprocessingConfig'] = TrajPreprocessingConfig
            libraries['TrajSegmentationConfig'] = TrajSegmentationConfig
            libraries['TrajPreprocessing'] = TrajPreprocessing

            from megabouts.segmentation import Segmentation
            libraries['Segmentation'] = Segmentation

            from megabouts.classification import TailBouts, BoutClassifier
            from megabouts.utils import bouts_category_color, bouts_category_name_short
            libraries['TailBouts'] = TailBouts
            libraries['BoutClassifier'] = BoutClassifier
            libraries['bouts_category_color'] = bouts_category_color
            libraries['bouts_category_name_short'] = bouts_category_name_short

            import torch
            import matplotlib.gridspec as gridspec
        except ImportError as e:
            print(f"Error importing required libraries: {e}")
            print("Please make sure all dependencies are installed.")
            sys.exit(1)

        # Get user inputs
        file_path = get_input_file_path()
        
        # Load data file
        print(f"\nLoading data from {file_path}...")
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            sys.exit(1)
            
        # Check if track column exists
        if 'track' not in df.columns:
            print("Error: CSV file must contain a 'track' column.")
            sys.exit(1)
            
        # Get available tracks
        available_tracks = df['track'].unique()
        if len(available_tracks) == 0:
            print("Error: No tracks found in the data.")
            sys.exit(1)
            
        # Get track selection from user
        selected_track_nums = get_track_selection(available_tracks)

        # Get certainty threshold
        certainty_threshold = get_certainty_threshold()
        
        # Get other parameters
        fps = get_fps()
        mm_per_unit = get_mm_per_unit()
        tail_points = get_tail_points()
        
        # Process each selected track
        results = {}
        errors = {}
        all_simplified_results = []
        
        for track_num in selected_track_nums:
            try:
                print(f"\n{'='*60}")
                print(f"=== Processing Track {track_num} ===")
                print(f"{'='*60}")
                
                bouts, error = process_track(
                    df, track_num, fps, mm_per_unit, tail_points, libraries
                )
                
                if error:
                    errors[track_num] = error
                    print(f"[Track {track_num}] Error: {error}")
                    continue
                    
                results[track_num] = bouts
                
                # Extract simplified results
                simplified_results = extract_simplified_results(
                    bouts.df, track_num, fps, libraries['bouts_category_name_short'], certainty_threshold
                )
                all_simplified_results.extend(simplified_results)
                
                # Print summary for this track
                print(f"\n--- Track {track_num} Analysis Results ---\n")
                
                bout_count = 0
                for result in simplified_results:
                    bout_count += 1
                    print(f"Bout #{bout_count} at onset {result['onset_frames']} ({result['onset_seconds']:.2f}s): "
                          f"Category {result['category']}, Certainty: {result['certainty']:.2f}")
                
                print(f"\n[Track {track_num}] Analysis complete. Found {bout_count} bouts with certainty > 0.5.")
                
            except Exception as e:
                errors[track_num] = str(e)
                print(f"[Track {track_num}] Error during processing: {e}")
        
        # Summary
        print(f"\n{'='*60}")
        print("=== Analysis Summary ===")
        print(f"{'='*60}")
        
        print(f"\nSuccessfully processed {len(results)} out of {len(selected_track_nums)} tracks.")
        print(f"Found {len(all_simplified_results)} total bouts with certainty > 0.5.")
        
        if errors:
            print("\nErrors encountered:")
            for track_num, error in errors.items():
                print(f"- Track {track_num}: {error}")
        
        # Ask if user wants to save results
        if all_simplified_results:
            save_results = input("\nDo you want to save the results to a CSV file? (y/n): ").lower()
            if save_results == 'y':
                output_file = input("Enter output filename (default: megabouts_results.csv): ").strip()
                if not output_file:
                    output_file = "megabouts_results.csv"
                if not output_file.endswith('.csv'):
                    output_file += '.csv'
                
                try:
                    # Create a DataFrame from the simplified results
                    results_df = pd.DataFrame(all_simplified_results)
                    
                    # Reorder columns to match requested format
                    ordered_columns = ['track', 'onset_frames', 'onset_seconds', 'category', 'certainty']
                    results_df = results_df[ordered_columns]
                    
                    # Rename columns to match requested format
                    column_mapping = {
                        'track': 'track',
                        'onset_frames': 'onset_frames',
                        'onset_seconds': 'onset_seconds',
                        'category': 'category',
                        'certainty': 'certainty'
                    }
                    results_df = results_df.rename(columns=column_mapping)
                    
                    # Round floating point columns
                    if 'onset_seconds' in results_df.columns:
                        results_df['onset_seconds'] = results_df['onset_seconds'].round(2)
                    if 'certainty' in results_df.columns:
                        results_df['certainty'] = results_df['certainty'].round(2)
                    
                    # Save to CSV
                    results_df.to_csv(output_file, index=False)
                    print(f"\nResults saved to {output_file} with simplified column format.")
                    
                except Exception as e:
                    print(f"Error saving results: {e}")
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()