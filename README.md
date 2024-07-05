# Detect Profanity in Surabaya Javanese Dialect

ini merupkakan sebuah projek Skripsi saya yang berjudul Deteksi Perkataan Vulgar Dalam Bahasa Jawa Dialek Surabaya Pada Konten Video Dengan Speech-To-Text

penelitian ini mengembangkan teknologi Speech-to-Text untuk mendeteksi perkataan vulgar dalam konten video di Indonesia, terutama perkataan vulgar pada dialek Suroboyoan. Hasil penelitian menunjukkan bahwa model yang telah di-finetunedengan data rekaman mandiri yang berisi kalimat-kalimat dalam Bahasa Jawa dialek Suroboyoan mampu mengenali perkataan vulgar dengan sangat baik. Model ini berhasil mencapai Word Error Rate (WER) sebesar 8%, yang menunjukkan performa yang sangat baik dalam mentranskripsi dan mendeteksi perkataan kasar. Selain itu, penelitian ini juga mengevaluasi efektivitas preprocessing audio seperti noise reduction dan volume normalization dalam menurunkan nilai WER. Hasil  eksperimen menunjukkan bahwa tahapan preprocessing ini berkontribusi signifikan  dalam meningkatkan akurasi transkripsi. Dengan demikian, penelitian ini memberikan kontribusi penting dalam upaya untuk memantau dan mengendalikan konten video di platform media, sekaligus menawarkan metodologi yang dapat digunakan untuk pengembangan model deteksi perkataan vulgar di masa depan.

# Model
model yang dikembangkan ini merupakan hasil fine-tune dari model [indonesian-nlp/wav2vec2-indonesian-javanese-sundanese](https://huggingface.co/indonesian-nlp/wav2vec2-indonesian-javanese-sundanese) menggunakan data [Profanity Speech Suroboyoan dataset](https://huggingface.co/datasets/Jaal047/profanity-speech-suroboyoan)

Saat menggunakan model ini, pastikan input ucapan Anda diambil sampelnya pada 16kHz.

## Cara Menggunakan
Model dapat digunakan secara langsung (tanpa model bahasa) sebagai berikut:
```python
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import noisereduce as nr
import librosa
import soundfile as sf

# Load model dan processor
processor = Wav2Vec2Processor.from_pretrained("Jaal047/profanity-javanese-sby")
model = Wav2Vec2ForCTC.from_pretrained("Jaal047/profanity-javanese-sby")

# Load dan kurangi noise dari audio
file_audio_path = 'audio.wav'
y, sr = librosa.load(file_audio_path, sr=16000)
reduced_noise = nr.reduce_noise(y=y, sr=sr)
sf.write('audio_reduced_noise1.wav', reduced_noise, sr)

# Fungsi untuk memuat dan preprocess audio
def load_and_preprocess_audio(file_path):
    audio_array, sampling_rate = torchaudio.load(file_path)
    if sampling_rate != 16000:
        audio_array = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)(audio_array)
    audio_array = torchaudio.transforms.Vol(gain=1.0, gain_type='amplitude')(audio_array)
    return audio_array.squeeze().numpy()

# Preprocess dan inferensi
audio_array = load_and_preprocess_audio('audio_reduced_noise1.wav')
inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
with torch.no_grad():
    logits = model(inputs.input_values).logits

# Ambil argmax dan decode prediksi
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]

print("Transkripsi:", transcription)
```
Model dapat dilihat [disini](https://huggingface.co/Jaal047/profanity-javanese-sby)

# Kesimpulan
Penelitian deteksi perkataan vulgar dalam Bahasa Jawa dialek Surabaya pada konten video menggunakan speech-to-text menghasilkan kesimpulan berikut:

1. **Peningkatan Performa Model**: Fine-tuning Model Multilingual Speech Recognition for Indonesian Languages dengan data perkataan vulgar dalam dialek Suroboyoan meningkatkan performa signifikan. Model mencapai Word Error Rate (WER) 8%, jauh di bawah target 15-20%, menunjukkan kemampuan tinggi dalam mengenali perkataan vulgar.

2. **Efektivitas Preprocessing**: Tahapan preprocessing seperti noise reduction dan volume normalization mengurangi WER hingga 15%. Model speech-to-text mampu mendeteksi dan mentranskripsi perkataan vulgar, termasuk dalam kondisi video dengan dua pembicara bersamaan. Analisis teks juga efektif menentukan waktu kemunculan kata-kata kasar dengan akurasi baik.

Secara keseluruhan, penelitian menunjukkan bahwa fine-tuning dengan data spesifik dan analisis teks tepat meningkatkan kemampuan model dalam mengenali, mentranskripsi, dan mengidentifikasi perkataan vulgar dalam dialek Suroboyoan, relevan untuk pengawasan konten media sosial dan platform video.

# Saran
1. **Perluasan Data**: Kumpulkan lebih banyak sampel audio dari berbagai sumber seperti YouTube, podcast, radio online, dan live streaming untuk meningkatkan adaptabilitas model terhadap variasi bahasa dan gangguan audio.

2. **Integrasi Model Speech Separation**: Pertimbangkan penggunaan model seperti Recurrent Neural Network Transducer (RNN-T) yang dimodifikasi untuk mengatasi overlapping speech, efektif dalam mengenali dan membedakan ucapan dua pembicara secara bersamaan.

3. **Penggunaan Model Bahasa**: Terapkan model bahasa untuk mengurangi WER dengan memperbaiki prediksi teks berdasarkan konteks dan struktur bahasa.

4. **Pengembangan Sistem Real-Time**: Kembangkan sistem monitoring dan filtering berbasis real-time untuk platform video dan media sosial, yang dapat memberikan peringatan atau tindakan otomatis terhadap konten yang mengandung kata-kata vulgar.
