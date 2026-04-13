# Smart Face Anti-Spoofing

Solusi ini dikembangkan untuk kompetisi Find IT UGM dengan fokus pada klasifikasi wajah anti-spoofing ke dalam 6 kelas.

## Ringkasan Kompetisi

Kompetisi ini berfokus pada pengembangan model yang mampu melakukan klasifikasi wajah ke dalam enam kelas berbeda secara akurat, sekaligus mengidentifikasi berbagai bentuk spoofing attack.

Tujuan utamanya adalah membangun model computer vision berbasis citra yang robust pada variasi kondisi nyata, seperti:

- perubahan pencahayaan
- variasi sudut pengambilan
- variasi tipe serangan spoofing

Model dievaluasi menggunakan Macro F1-Score, yaitu rata-rata F1 dari seluruh kelas, sehingga setiap kelas memiliki kontribusi yang sama meskipun distribusi data tidak seimbang.

## Kelas pada Dataset

- realperson: citra wajah asli tanpa manipulasi
- fake_printed: serangan menggunakan foto cetak
- fake_screen: serangan menggunakan tampilan layar perangkat digital
- fake_mask: serangan menggunakan masker wajah
- fake_mannequin: serangan menggunakan replika/mannequin
- fake_unknown: jenis spoofing lain di luar kategori utama

## Pendekatan Solusi

Pipeline utama ada di notebook berikut:

- notebook/face-spoofing-newest.ipynb

Ringkasan pendekatan:

1. Data Preparation
- Mengunduh dataset, sample submission, dan best model dari Google Drive.
- Mengekstrak dataset dan memvalidasi struktur folder.
- Menyusun pasangan data wajah crop (YOLO) dan gambar full face.

2. Preprocessing dan Augmentasi
- Transform utama menggunakan AutoImageProcessor dari model backbone.
- Augmentasi: RandomResizedCrop, RandomHorizontalFlip, ColorJitter, GaussianBlur, grayscale, JPEG compression, low-resolution simulation, dan Gaussian noise.
- Strategi training mencampur image crop wajah dan image full untuk memperkuat generalisasi.

3. Model
- Backbone: facebook/dinov3-convnext-large-pretrain-lvd1689m.
- Head klasifikasi custom dengan adapter residual + classifier MLP.
- Loss: Focal Loss untuk membantu penanganan distribusi kelas yang berpotensi tidak seimbang.

4. Training
- Trainer dari Hugging Face Transformers.
- Early stopping aktif.
- Model terbaik dipilih berdasarkan macro_f1.

5. Evaluasi
- Evaluasi validasi dengan accuracy dan macro_f1.
- Classification report.
- Confusion matrix.
- Visualisasi contoh prediksi yang salah (misclassified).

6. Inference dan Submission
- Inference menggunakan best model (hasil training terbaik sebelumnya).
- Test-time augmentation multi-scale + horizontal flip.
- Fusion probabilitas antara gambar crop wajah dan gambar full.
- Output akhir: file submission CSV.

## Struktur Repository

- notebook/face-spoofing-newest.ipynb
- assets/images/
- README.md

## Menambahkan Gambar ke README

Folder gambar sudah disiapkan di:

- assets/images/

Silakan export atau screenshot output penting dari notebook, lalu paste ke folder tersebut dengan nama file berikut agar otomatis terhubung ke section ini:

- assets/images/label-distribution.png
- assets/images/sample-images.png
- assets/images/confusion-matrix.png
- assets/images/misclassified-examples.png

Contoh visualisasi:

### Distribusi Label

![Label Distribution](assets/images/label-distribution.png)

### Contoh Gambar per Kelas

![Sample Images](assets/images/sample-images.png)

### Confusion Matrix

![Confusion Matrix](assets/images/confusion-matrix.png)

### Misclassified Examples

![Misclassified Examples](assets/images/misclassified-examples.png)

## Cara Menjalankan Notebook

Notebook ini disusun untuk environment Kaggle (terlihat dari path /kaggle/working). Jalankan sel secara berurutan dari awal sampai akhir.

Catatan:

- Pastikan koneksi internet aktif saat mengunduh dataset/model dari Google Drive.
- Jika ingin menjalankan lokal, sesuaikan path direktori data/model.

## Hasil Output Utama

File output utama dari pipeline:

- Model weights (.pth)
- Submission CSV untuk kompetisi

Silakan tambahkan skor eksperimen terbaik Anda di bawah ini:

- Best Validation Macro F1: isi di sini
- Public/Private Leaderboard Score: isi di sini

## Kredit

Dikembangkan sebagai solusi kompetisi Find IT UGM oleh tim proyek ini.
