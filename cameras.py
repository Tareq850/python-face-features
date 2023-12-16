import cv2
import keyboard
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import tkinter as tk
import time

# تحميل نموذج مسبق تدريبه لتصنيف المزاج
model_path = 'C:/Users/UX/.spyder-py3/smile.keras'  # قم بتوفير مسار النموذج
emotion_model = load_model(model_path, compile=False)

# تحميل ملف Haarcascades للوجوه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

expected_image_size = (224, 224)
# قائمة تصنيفات المزاج
emotion_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']

# فتح كاميرا الويب
cap = cv2.VideoCapture(0)
frame_delay = 0.8

# إنشاء نافذة tkinter
root = tk.Tk()
root.title("Emotion Detection")

# إنشاء عنصر نص لعرض حالة النفسية
emotion_label = tk.Label(root, text="Emotion: ", font=("Helvetica", 16))
emotion_label.pack()

def update_emotion_label(emotion):
    emotion_label.config(text=f"Emotion: {emotion}")

# دالة لتحديث حالة النفسية في واجهة المستخدم
def update_emotion_gui(emotion):
    root.after(100, update_emotion_label, emotion)

while True:
    # قراءة الإطار من الكاميرا
    ret, frame = cap.read()

    # تحويل الإطار إلى الأبيض والأسود (الرمادي) لتحسين أداء تحديد الوجوه
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # تحديد الوجوه في الإطار
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # رسم مربعات حول الوجوه وتصنيف المزاج
    for (x, y, w, h) in faces:
        # قص الوجه من الإطار وتغيير حجمه إلى (64, 64)
        face_roi = cv2.resize(gray[y:y+h, x:x+w], expected_image_size)

        # تكرار القنوات الرمادية لتكون ثلاث قنوات
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2RGB)

        # تحويل الوجه إلى مصفوفة
        face_roi = img_to_array(face_roi)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0  # تقليل قيم البكسل لتوافق نموذج المزاج

        # التنبؤ باستخدام نموذج المزاج
        emotion_probabilities = emotion_model.predict(face_roi)[0]
        emotion_label = emotion_labels[np.argmax(emotion_probabilities)]

        # رسم مربع حول الوجه
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # عرض تصنيف المزاج على الشاشة
        cv2.putText(frame, f'Emotion: {emotion_label}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # تحديث حالة النفسية في واجهة المستخدم
        update_emotion_gui(emotion_label)

    # عرض الإطار المعدل
    cv2.imshow('Emotion Detection', frame)

    # انتظار لضغط مفتاح لإيقاف البرنامج (ESC)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # انتظار لضغط مفتاح لإيقاف الكاميرا (مثلاً، حرف "q")
    if keyboard.is_pressed('q'):
        break

# إغلاق كاميرا الويب وتدمير النوافذ
cap.release()
cv2.destroyAllWindows()

# تشغيل دورة الحدث في واجهة المستخدم
root.mainloop()
