import torch
import whisper
PATH = "D:\Ivan\STT_Test\stevetest1.mp3"

model = whisper.load_model("base")
result = model.transcribe(PATH,fp16=False)
print(result["text"])