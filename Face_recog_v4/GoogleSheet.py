# from google.oauth2.credentials import Credentials
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# import os.path

# # Scope defines access level
# SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# SPREADSHEET_ID = 'YOUR_SPREADSHEET_ID_HERE'
# RANGE_NAME = 'Sheet1!A1:D5'

# def main():
#     creds = None

#     # Token stores user access after first login
#     if os.path.exists('token.json'):
#         creds = Credentials.from_authorized_user_file('token.json', SCOPES)

#     # First-time authentication
#     if not creds or not creds.valid:
#         flow = InstalledAppFlow.from_client_secrets_file(
#             'credentials.json', SCOPES
#         )
#         creds = flow.run_local_server(port=0)

#         with open('token.json', 'w') as token:
#             token.write(creds.to_json())

#     service = build('sheets', 'v4', credentials=creds)
#     sheet = service.spreadsheets()

#     # 🔹 READ DATA
#     result = sheet.values().get(
#         spreadsheetId=SPREADSHEET_ID,
#         range=RANGE_NAME
#     ).execute()

#     values = result.get('values', [])
#     print("Current Data:")
#     for row in values:
#         print(row)

#     # 🔹 WRITE DATA
#     new_values = [
#         ["Name", "Age", "City"],
#         ["Alice", "25", "New York"],
#         ["Bob", "30", "London"]
#     ]

#     sheet.values().update(
#         spreadsheetId=SPREADSHEET_ID,
#         range="Sheet1!A1",
#         valueInputOption="RAW",
#         body={"values": new_values}
#     ).execute()

#     print("Sheet updated successfully!")

# if __name__ == '__main__':
#     main()


# # spread sheet 
# # https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit


import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import calendar
import os

# ---------------- CONFIG ----------------
SPREADSHEET_NAME = "Attendance"
DATASET_PATH = "Dataset/People"
CREDENTIALS_FILE = "credentials.json"
# ----------------------------------------

# Google auth
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
client = gspread.authorize(creds)

spreadsheet = client.open(SPREADSHEET_NAME)

# ------------------------------------------------
def get_students():
    """Get student names from Dataset/People folders"""
    return sorted([
        name for name in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, name))
    ])

# ------------------------------------------------
def get_month_sheet_name(date=None):
    date = date or datetime.now()
    return date.strftime("%B %Y")  # e.g. "September 2026"

# ------------------------------------------------
def get_or_create_month_sheet():
    today = datetime.now()
    sheet_name = get_month_sheet_name(today)

    try:
        sheet = spreadsheet.worksheet(sheet_name)
        return sheet
    except gspread.WorksheetNotFound:
        return setup_month_sheet(sheet_name, today)

# ------------------------------------------------
def setup_month_sheet(sheet_name, date):
    print(f"Creating new sheet: {sheet_name}")

    sheet = spreadsheet.add_worksheet(
        title=sheet_name,
        rows=100,
        cols=40
    )

    students = get_students()
    days_in_month = calendar.monthrange(date.year, date.month)[1]

    # Header row (dates)
    header = ["Name"] + [str(day) for day in range(1, days_in_month + 1)]
    sheet.insert_row(header, 1)

    # Student rows
    for i, student in enumerate(students, start=2):
        sheet.update_cell(i, 1, student)

    return sheet

# ------------------------------------------------
def mark_attendance(student_name):
    print("HELLO")
    # sheet = get_or_create_month_sheet()

    # today = datetime.now()
    # day_column = today.day + 1  # +1 because column 1 is "Name"

    # # Get all names from column A
    # names = sheet.col_values(1)

    # if student_name not in names:
    #     print(f"Student '{student_name}' not found")
    #     return

    # row = names.index(student_name) + 1

    # sheet.update_cell(row, day_column, "P")
    # print(f"Marked {student_name} present on {today.strftime('%d %B')}")

# ------------------------------------------------
if __name__ == "__main__":
    # Example usage
    mark_attendance("KUNAL")
