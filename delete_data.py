import os
import shutil
import json
import glob

# --- Define file and folder paths ---
DATASET_PATH = 'dataset'
TRAINER_FILE = 'trainer/trainer.yml'
FACE_DB_FILE = 'face_database.pkl'
MAP_FILE = 'id_to_name_map.json'
ATTENDANCE_LOG = 'attendance_log.csv'
ATTENDANCE_SUMMARY = 'attendance_summary.csv'

def clear_all_data():
    """Deletes all generated data, folders, and models."""
    print("\n--- CLEARING ALL DATA ---")
    
    # 1. Delete dataset folder
    if os.path.exists(DATASET_PATH):
        try:
            shutil.rmtree(DATASET_PATH)
            print(f"✅ Removed folder: {DATASET_PATH}")
        except Exception as e:
            print(f"❌ Error removing {DATASET_PATH}: {e}")
    else:
        print(f"ℹ️ Folder not found: {DATASET_PATH}")

    # 2. Delete trainer file
    if os.path.exists(TRAINER_FILE):
        try:
            os.remove(TRAINER_FILE)
            print(f"✅ Removed file: {TRAINER_FILE}")
        except Exception as e:
            print(f"❌ Error removing {TRAINER_FILE}: {e}")
    else:
        print(f"ℹ️ File not found: {TRAINER_FILE}")

    # 3. Delete face database
    if os.path.exists(FACE_DB_FILE):
        try:
            os.remove(FACE_DB_FILE)
            print(f"✅ Removed file: {FACE_DB_FILE}")
        except Exception as e:
            print(f"❌ Error removing {FACE_DB_FILE}: {e}")
    else:
        print(f"ℹ️ File not found: {FACE_DB_FILE}")

    # 4. Delete ID map
    if os.path.exists(MAP_FILE):
        try:
            os.remove(MAP_FILE)
            print(f"✅ Removed file: {MAP_FILE}")
        except Exception as e:
            print(f"❌ Error removing {MAP_FILE}: {e}")
    else:
        print(f"ℹ️ File not found: {MAP_FILE}")

    # 5. Delete logs
    if os.path.exists(ATTENDANCE_LOG):
        try:
            os.remove(ATTENDANCE_LOG)
            print(f"✅ Removed file: {ATTENDANCE_LOG}")
        except Exception as e:
            print(f"❌ Error removing {ATTENDANCE_LOG}: {e}")
    
    if os.path.exists(ATTENDANCE_SUMMARY):
        try:
            os.remove(ATTENDANCE_SUMMARY)
            print(f"✅ Removed file: {ATTENDANCE_SUMMARY}")
        except Exception as e:
            print(f"❌ Error removing {ATTENDANCE_SUMMARY}: {e}")

    print("\nAll data has been cleared.")

def clear_specific_user():
    """Deletes all data related to one specific user."""
    print("\n--- CLEARING SPECIFIC USER ---")
    
    name_and_id = input("Enter the Name_ID (e.g., Jit_101): ").strip()
    numeric_id = input(f"Enter the corresponding Numeric ID (e.g., 1): ").strip()

    if not name_and_id or not numeric_id:
        print("Both inputs are required. Aborting.")
        return

    # 1. Delete user's subfolder (from enroll_user.py / combined_enrollment.py)
    user_folder_path = os.path.join(DATASET_PATH, name_and_id)
    if os.path.exists(user_folder_path):
        try:
            shutil.rmtree(user_folder_path)
            print(f"✅ Removed folder: {user_folder_path}")
        except Exception as e:
            print(f"❌ Error removing {user_folder_path}: {e}")
    else:
        print(f"ℹ️ Folder not found: {user_folder_path}")

    # 2. Delete user's numbered images (from face_dataset.py / combined_enrollment.py)
    # This finds files like "dataset/User.1.1.jpg", "dataset/User.1.2.jpg", etc.
    image_pattern = os.path.join(DATASET_PATH, f"User.{numeric_id}.*.jpg")
    user_images = glob.glob(image_pattern)
    
    if user_images:
        count = 0
        for img_path in user_images:
            try:
                os.remove(img_path)
                count += 1
            except Exception as e:
                print(f"❌ Error removing {img_path}: {e}")
        print(f"✅ Removed {count} images matching '{image_pattern}'")
    else:
        print(f"ℹ️ No images found matching: {image_pattern}")

    # 3. Remove user from id_to_name_map.json
    if os.path.exists(MAP_FILE):
        try:
            with open(MAP_FILE, 'r') as f:
                id_to_name_map = json.load(f)
            
            if numeric_id in id_to_name_map:
                removed_name = id_to_name_map.pop(numeric_id)
                print(f"✅ Removed '{removed_name}' (ID: {numeric_id}) from {MAP_FILE}")
                
                # Save the updated map back to the file
                with open(MAP_FILE, 'w') as f:
                    json.dump(id_to_name_map, f, indent=4)
            else:
                print(f"ℹ️ Numeric ID {numeric_id} not found in {MAP_FILE}")
        
        except Exception as e:
            print(f"❌ Error updating {MAP_FILE}: {e}")
    else:
        print(f"ℹ️ Map file not found: {MAP_FILE}")

    print("\n" + "="*40)
    print("⚠️ IMPORTANT ⚠️")
    print("User data has been deleted from the dataset.")
    print("You MUST run 'train_model.py' and 'encode_face.py' again")
    print("to update your models before running the system!")
    print("="*40)

def main():
    """Main menu for the data management script."""
    while True:
        print("\n=== Data Management Utility ===")
        print("1. Clear ALL data (Deletes all datasets, models, and logs)")
        print("2. Clear data for a SPECIFIC user")
        print("q. Quit")
        
        choice = input("Enter your choice (1, 2, or q): ").strip().lower()
        
        if choice == '1':
            confirm = input("ARE YOU SURE you want to delete ALL data? This cannot be undone. (yes/no): ").strip().lower()
            if confirm == 'yes':
                clear_all_data()
            else:
                print("Aborted.")
        
        elif choice == '2':
            clear_specific_user()
            
        elif choice == 'q':
            print("Exiting.")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()