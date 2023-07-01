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
            "name": "Linus Torvald",
            "detection_no": 0

        },
    "02":
        {
            "name": "Ana De Armas",
            "detection_no": 0
        },
    "03":
        {
            "name": "Elon Musk",
            "detection_no": 0
        },
    "04":
        {
            "name": "Emily Blunt",
            "detection_no": 0
        },
    "05":
        {
            "name": "",
            "detection_no": 0
        },
    "06":
        {
            "name": "",
            "detection_no": 0
        },
    "07":
        {
            "name": "",
            "detection_no": 0
        },
    "08":
        {
            "name": "",
            "detection_no": 0
        },
    "09":
        {
            "name": "",
            "detection_no": 0
        },
    "10":
        {
            "name": "",
            "detection_no": 0
        },
    "11":
        {
            "name": "",
            "detection_no": 0
        },
    "12":
        {
            "name": "",
            "detection_no": 0
        },
    "13":
        {
            "name": "",
            "detection_no": 0
        },
    "14":
        {
            "name": "",
            "detection_no": 0
        },
    "15":
        {
            "name": "",
            "detection_no": 0
        },
    "16":
        {
            "name": "",
            "detection_no": 0
        },
    "17":
        {
            "name": "",
            "detection_no": 0
        },
    "18":
        {
            "name": "",
            "detection_no": 0
        },
    "19":
        {
            "name": "",
            "detection_no": 0
        },
    "20":
        {
            "name": "",
            "detection_no": 0
        },

}

for key, value in data.items():
    ref.child(key).set(value)
