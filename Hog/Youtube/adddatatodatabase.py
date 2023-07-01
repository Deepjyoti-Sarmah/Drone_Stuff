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
        }
}

for key, value in data.items():
    ref.child(key).set(value)
