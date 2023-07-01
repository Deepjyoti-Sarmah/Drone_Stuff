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
            "name": "Linus Torvald"

        },
    "02":
        {
            "name": "Ana De Armas"
        },
    "03":
        {
            "name": "Elon Musk"
        },
    "04":
        {
            "name": "Emily Blunt"
        }
}

for key, value in data.items():
    ref.child(key).set(value)
