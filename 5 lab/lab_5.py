import pyaudio
import numpy as np
import whisper
import webbrowser

model = whisper.load_model("base")

def recognize_speech():
    record_duration = 1
    print("Говорите")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=44100)
    frames = []

    for _ in range(int(44100/1024 * record_duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.int16))
    # Hello

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio = np.concatenate(frames)
    audio = whisper.pad_or_trim(audio)
    audio = audio.astype(np.float32)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")
    # Detected language: en
    options = whisper.DecodingOptions()
    command = whisper.decode(model, mel, options)

    print(command.text)
    # You
    return command.text

def open_web(command):
    if "Yandex" in command:
        search_query = command.replace("поиск в яндексе", "").strip()
        webbrowser.open(f"https://yandex.ru/search/?text={command}")
    elif "Вконтакте" in command:
        webbrowser.open("https://vk.com/")
    elif "Youtube" in command:
        search_query = command.replace("найти видео в rutube", "").strip()
        webbrowser.open(f"https://rutube.ru/search/?query=={search_query}")
    else:
        print("Команда не распознана.")
    

command = recognize_speech().lower()
print(command)
open_web(command)
# Команда не распознана.

