import sys
import speech_recognition as sr

path = sys.argv[1]

r = sr.Recognizer()

aud  =sr.AudioFile(path)
with aud as source:
    audio = r.record(source)
try:
    s = r.recognize_google(audio)
    print("Text: "+s)
except Exception as e:
    print("Exception: "+str(e))

sys.stdout.flush()