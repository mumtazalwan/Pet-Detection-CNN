# Deep Learning: Analisis Konseptual & Klasifikasi Ras Hewan Peliharaan (Oxford-IIIT Pet)

Repository ini berisi dokumentasi lengkap dari notebook Colab yang mencakup **Analisis Konseptual Deep Learning** serta **Eksperimen Praktik** menggunakan PyTorch untuk klasifikasi ras hewan menggunakan transfer learning[cite: 28].

---

## Daftar Isi
1. [Bagian A: Analisis Konseptual](#bagian-a-analisis-konseptual)
2. [Bagian B: Eksperimen Praktik & Konfigurasi](#bagian-b-eksperimen-praktik--konfigurasi)
3. [Tech Stack & Library](#tech-stack--library)

---

## Bagian A: Analisis Konseptual

Bagian ini membahas pemahaman mendalam mengenai teori-teori dasar dalam arsitektur CNN dan pelatihan model[cite: 28]:
* **Efek Regularisasi pada Batch Normalization**: Normalisasi aktivasi menggunakan rata-rata dan varians mini-batch yang dinamis menimbulkan efek bising acak, bertindak sebagai *mild regularization*[cite: 28].
* **Degradasi Jaringan & Skip Connections**: Arsitektur ResNet menggunakan formulasi $y = F(x) + x$ untuk menambahkan suku identitas pada *backpropagation*, sehingga mengatasi masalah *vanishing gradient* pada jaringan dalam[cite: 28].
* **Feature Extraction vs. Fine-Tuning**: Feature Extraction membekukan *backbone* (cocok untuk data kecil), sedangkan Fine-Tuning membuka lapisan jaringan untuk performa lebih tinggi dengan risiko *destructive gradient updates* jika tidak dikontrol dengan baik[cite: 28].
* **Kaidah Data Augmentation**: Pipeline augmentasi wajib diaktifkan hanya pada *train set* dan dimatikan pada *val/test set* agar evaluasi objektif. Teknik seperti *vertical flip* tidak layak untuk klasifikasi ras hewan karena melanggar invariansi semantik[cite: 28].
* **Mitigasi Catastrophic Forgetting**: Mencegah rusaknya representasi fitur bawaan model *pretrained* menggunakan strategi seperti *differential learning rate* dan *gradual unfreezing*[cite: 28].

---

## Bagian B: Eksperimen Praktik & Konfigurasi

### Konfigurasi Hyperparameter
Eksperimen ini berjalan dengan pengaturan parameter berikut[cite: 28]:
* **Seed**: `42` (untuk menjamin reproduksibilitas hasil)[cite: 28]
* **Ukuran Gambar**: `224 x 224` piksel[cite: 28]
* **Batch Size**: `32`[cite: 28]
* **Epoch Maksimal**: `30` (dengan mekanisme *early stopping* *patience* `5`)[cite: 28]
* **Pembagian Dataset**: Stratified split dengan rasio **70% Train / 15% Validation / 15% Test**[cite: 28]

### Dataset Oxford-IIIT Pet
* Mengambil data langsung dari `torchvision.datasets.OxfordIIITPet`[cite: 28].
* Difokuskan pada **10 kelas terpilih** (5 ras kucing dan 5 ras anjing) dengan total **1.987 gambar**[cite: 28].
* Menggunakan metrik evaluasi **Macro F1-Score** untuk menangani sedikit perbedaan jumlah sampel antar kelas (misalnya *Abyssinian* dan *Staffordshire*)[cite: 28].

---

## Tech Stack & Library
Proyek ini dijalankan menggunakan lingkungan Python dengan *dependency* utama[cite: 28]:
* **PyTorch** & **Torchvision**[cite: 28]
* **Scikit-learn**[cite: 28]
* **NumPy** & **Pandas**[cite: 28]
* **Matplotlib**[cite: 28]