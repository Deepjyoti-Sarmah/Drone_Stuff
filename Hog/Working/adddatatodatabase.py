import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-recognition-june-2023-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "01":
        {
            "name": "HariOm Kr Modi",
            "detection_no": 0

        },
    "02":
        {
            "name": "Ranjeet_kumar_gupta",
            "detection_no": 0
        },
    "03":
        {
            "name": "Doni_Khungev_Basumatary",
            "detection_no": 0
        },
    "04":
        {
            "name": "Prince Raj Kumar",
            "detection_no": 0
        },
    "05":
        {
            "name": "Mayuk_Bhattachargy",
            "detection_no": 0
        },
    "06":
        {
            "name": "Saif_Ali",
            "detection_no": 0
        },
    "07":
        {
            "name": "Rohit Kumar",
            "detection_no": 0
        },
    "08":
        {
            "name": "Pratynsh_J_Deka",
            "detection_no": 0
        },
    "09":
        {
            "name": "Saurabh_kumar",
            "detection_no": 0
        },
    "10":
        {
            "name": "Rahul Thakur",
            "detection_no": 0
        },
    "11":
        {
            "name": "Saksam Som",
            "detection_no": 0
        },
    "12":
        {
            "name": "Rahul Kr Sharma",
            "detection_no": 0
        },
    "13":
        {
            "name": "Vikrant Kumar",
            "detection_no": 0
        },
    "14":
        {
            "name": "Santanu Ghose",
            "detection_no": 0
        },
    "15":
        {
            "name": "Nabojyoti Nath",
            "detection_no": 0
        },
    "16":
        {
            "name": "Ravi Raj",
            "detection_no": 0
        },
    "17":
        {
            "name": "Anand Raj",
            "detection_no": 0
        },
    "18":
        {
            "name": "Ramavtar Toshi",
            "detection_no": 0
        },
    "19":
        {
            "name": "Rishav_Sheivastaur",
            "detection_no": 0
        },
    "20":
        {
            "name": "Md Aziz",
            "detection_no": 0
        },

}

for key, value in data.items():
    ref.child(key).set(value)
