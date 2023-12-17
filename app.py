from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = tf.keras.models.load_model('model_gaji.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/klasifikasi', methods=['GET'])
def klasifikasi():
    return render_template('klasifikasi.html', output=None)

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [str(x) for x in request.form.values()]

    umur = int(int_features[0])
    thn_pendidikan = int(int_features[1])
    berat_akhir = int(int_features[2])
    kelas_pekerja = int(int_features[3])
    status_perkawinan = int(int_features[4])
    pekerjaan = int(int_features[5])
    nilai_aset = int(int_features[6])
    jam_kerja = int(int_features[7])

    # make a list
    raw = [umur, berat_akhir, thn_pendidikan, nilai_aset, jam_kerja]

    # add kelas pekerja
    for i in range(9):
        if i != 0:
            if i == kelas_pekerja:
                raw.append(1)
            else:
                raw.append(0)

    # add status perkawinan
    for i in range(7):
        if i != 0:
            if i == kelas_pekerja:
                raw.append(1)
            else:
                raw.append(0)

    # add pekerjaan
    for i in range(16):
        if i != 0:
            if i == kelas_pekerja:
                raw.append(1)
            else:
                raw.append(0)

    dt = pd.DataFrame([raw], columns=["Umur", "Berat Akhir", "Jmlh Tahun Pendidikan", "Keuntungan Kapital", "Jam per Minggu", "Kelas Pekerja_Pekerja Bebas Bukan Perusahan", "Kelas Pekerja_Pekerja Bebas Perusahaan", "Kelas Pekerja_Pemerintah Lokal", "Kelas Pekerja_Pemerintah Negara", "Kelas Pekerja_Pemerintah Provinsi", "Kelas Pekerja_Tanpa di Bayar", "Kelas Pekerja_Tidak Pernah Bekerja", "Kelas Pekerja_Wiraswasta", "Status Perkawinan_Belum Pernah Menikah", "Status Perkawinan_Berpisah", "Status Perkawinan_Cerai", "Status Perkawinan_Janda", "Status Perkawinan_Menikah", "Status Perkawinan_Menikah LDR", "Pekerjaan_Asisten Rumah Tangga", "Pekerjaan_Ekesekutif Managerial", "Pekerjaan_Mesin Inspeksi", "Pekerjaan_Pekerjaan Lainnya", "Pekerjaan_Pembersih", "Pekerjaan_Pemuka Agama", "Pekerjaan_Penjaga", "Pekerjaan_Perbaikan Kerajinan", "Pekerjaan_Petani", "Pekerjaan_Sales", "Pekerjaan_Servis Lainnya", "Pekerjaan_Spesialis", "Pekerjaan_Supir", "Pekerjaan_Tech-support", "Pekerjaan_Tentara"])

    # scaling
    scaler = StandardScaler()
    numerical_features = ['Umur', 'Berat Akhir', 'Jmlh Tahun Pendidikan', 'Keuntungan Kapital', 'Jam per Minggu']

    csv = pd.read_csv('train.csv')
    train = csv[numerical_features]
    user_input = dt[numerical_features]
    train_merged = pd.concat([train, user_input], axis=0)

    # scaling
    scall_result = scaler.fit_transform(train_merged)
    dt[numerical_features] = scall_result[-1:]

    # # predict
    prediction = model.predict(dt)
    print(prediction)

    # apakah gaji <= 7jt
    if prediction[0][0] > prediction[0][1]:
        output = True
    else:
        output = False

    return render_template('klasifikasi.html', output=output)

if __name__ == "__main__":
    app.run(debug=True)