import pandas as pd
from datetime import datetime
import os

def view_attendance():
    if not os.path.exists('attendance_log.csv'):
        print("No attendance records found!")
        return
    
    # Read the CSV file
    df = pd.read_csv('attendance_log.csv')
    
    print("=== ATTENDANCE REPORT ===")
    print(f"Total records: {len(df)}")
    print("\nRecent entries:")
    print(df.tail(10))  # Show last 10 entries
    
    # Summary by student
    print("\n=== SUMMARY BY STUDENT ===")
    student_summary = df.groupby(['Student_ID', 'Student_Name', 'Action']).size().unstack(fill_value=0)
    print(student_summary)
    
    # Today's attendance
    today = datetime.now().strftime("%Y-%m-%d")
    today_data = df[df['Date'] == today]
    if not today_data.empty:
        print(f"\n=== TODAY'S ATTENDANCE ({today}) ===")
        print(today_data)
    
    # Save summary to file
    student_summary.to_csv('attendance_summary.csv')
    print(f"\nSummary saved to 'attendance_summary.csv'")

if __name__ == "__main__":
    view_attendance()
