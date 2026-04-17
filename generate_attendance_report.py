import os
import pandas as pd

# Path to the attendance folder
attendance_folder = '../attendance'

# List to store all attendance data
all_attendance_data = []

# Traverse through all files in the attendance folder
for filename in os.listdir(attendance_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(attendance_folder, filename)

        try:
            # Read each CSV into a DataFrame
            df = pd.read_csv(file_path)

            # Extract the date from the filename (assuming the filename is YYYY-MM-DD.csv)
            date = os.path.splitext(filename)[0]

            # Add a 'Date' column to the DataFrame
            df['Date'] = date

            # Append the DataFrame to the list
            all_attendance_data.append(df)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Combine all DataFrames into one
if all_attendance_data:
    combined_attendance_df = pd.concat(all_attendance_data, ignore_index=True)

    # Rearrange columns to 'Date', 'Name', 'Time'
    combined_attendance_df = combined_attendance_df[['Date', 'Name', 'Time']]

    # Display the combined attendance data
    print("\nCombined Attendance Data:")
    print(combined_attendance_df)

    # Optionally, save combined attendance to a new CSV
    combined_attendance_file = '../attendance/combined_attendance.csv'
    combined_attendance_df.to_csv(combined_attendance_file, index=False)
    print(f"\nCombined attendance saved to: {combined_attendance_file}")

else:
    print("No attendance data found.")