import torch
import whisper

#must check this for better model versions, whisper related
model = whisper.load_model("base")

def getTextFromAudio(PATH):  
  result = model.transcribe(PATH,fp16=False)
  return result["text"] 
