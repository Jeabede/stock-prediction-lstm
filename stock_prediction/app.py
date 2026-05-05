import matplotlib
matplotlib.use('Agg')  # Non-interactive backend untuk web server
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, send_file, url_for, redirect
import pandas as pd
import yfinance as yf
import os
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import base64
import joblib
import json
import shutil
from werkzeug.utils import secure_filename

import warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from urllib.parse import quote
from scraper import scrape_yahoo_finance

def toast_redirect(endpoint, message, toast_type='error'):
    """Redirect ke halaman dengan popup toast notification"""
    return redirect(url_for(endpoint, toast_msg=quote(message), toast_type=toast_type))

app = Flask(__name__)
app.secret_key = "jquery42"

# Folder penyimpanan model dan data
MODEL_DIR = "models"
DATA_DIR = "data"
UPLOAD_DIR = "uploads"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Variabel global untuk menyimpan data
scalers = {}
train_data, test_data = None, None
time_step = 5  # Window size untuk LSTM
stock_name = ''
start_date = ''
end_date = ''

# Fungsi untuk membuat dataset dalam bentuk sequence
def create_dataset(data, time_step):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        Y.append(data[i+time_step])
    return np.array(X), np.array(Y)

def calculate_errors(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    actual_trend = np.where(np.diff(np.append([actual[0]], actual)) > 0, 1, 0)
    pred_trend = np.where(np.diff(np.append([predicted[0]], predicted)) > 0, 1, 0)

    min_len = min(len(actual_trend), len(pred_trend))
    f1 = f1_score(actual_trend[:min_len], pred_trend[:min_len], average='binary')
    
    return {"MSE": round(mse, 4), "RMSE": round(rmse, 4), "MAE": round(mae, 4), "F1": round(f1, 4)}

# Fungsi baru untuk menganalisis tren harga saham
def analyze_price_trend(actual_close, predicted_close):
    # Analisis tren berdasarkan beberapa nilai terakhir dari data prediksi
    last_days = 5  # Menggunakan 5 hari terakhir untuk analisis tren
    if len(predicted_close) < last_days + 1:
        last_days = len(predicted_close) - 1
        if last_days < 1:  # Jika data terlalu sedikit
            last_days = 1
            
    # Analisis tren pada data prediksi sendiri
    first_pred = predicted_close[-last_days]
    last_pred = predicted_close[-1]
    
    # Tentukan tren berdasarkan perubahan nilai prediksi
    if last_pred > first_pred:
        trend = "NAIK"
        percentage = ((last_pred - first_pred) / first_pred) * 100
    elif last_pred < first_pred:
        trend = "TURUN"
        percentage = ((first_pred - last_pred) / first_pred) * 100
    else:
        trend = "STABIL"
        percentage = 0
    
    # Format output dengan 2 desimal
    formatted_first = "{:.2f}".format(first_pred)
    formatted_last = "{:.2f}".format(last_pred)
    
    # Buat pesan analisis tren
    trend_message = f"Analisis Tren Harga Saham: \nBerdasarkan analisis {last_days} hari terakhir, harga saham diperkirakan akan {trend} dari {formatted_first} ke {formatted_last} (perubahan {percentage:.2f}%)."
    
    return trend_message, trend

def cleanup_model_files():
    """Fungsi untuk menghapus file model dan data terkait setelah digunakan"""
    files_deleted = []
    errors = []
    
    try:
        # Pastikan stock_name tersedia
        global stock_name
        
        # Daftar file yang perlu dihapus
        files_to_delete = [
            # Model files
            os.path.join(MODEL_DIR, "lstm_model.h5"),
            os.path.join(MODEL_DIR, "scalers.pkl"),
            os.path.join(MODEL_DIR, "model_params.json"),
            # Data files
            os.path.join(DATA_DIR, "df.json"),
            os.path.join(DATA_DIR, "train_data.json"),
            os.path.join(DATA_DIR, "test_data.json"),
            os.path.join(DATA_DIR, "model_params.json"),
            # Static files
            os.path.join("static", "predictions.csv")
        ]
        
        # Tambahkan file CSV saham jika ada
        if stock_name:
            files_to_delete.append(os.path.join("static", f"{stock_name}_data.csv"))
            
        # Coba hapus satu per satu dengan error handling
        for file_path in files_to_delete:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    files_deleted.append(file_path)
                    print(f"✅ File berhasil dihapus: {file_path}")
            except Exception as e:
                errors.append(f"❌ Gagal menghapus {file_path}: {str(e)}")
                print(f"❌ Error: {str(e)}")
        
        # Hapus paket model
        try:
            model_package_dir = os.path.join(MODEL_DIR, "model_package")
            if os.path.exists(model_package_dir):
                shutil.rmtree(model_package_dir)
                files_deleted.append(model_package_dir)
                print(f"✅ Directory berhasil dihapus: {model_package_dir}")
        except Exception as e:
            errors.append(f"❌ Gagal menghapus direktori model_package: {str(e)}")
            print(f"❌ Error: {str(e)}")
        
        # Hapus file zip di MODEL_DIR
        try:
            zip_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.zip')]
            for zip_file in zip_files:
                zip_path = os.path.join(MODEL_DIR, zip_file)
                try:
                    os.remove(zip_path)
                    files_deleted.append(zip_path)
                    print(f"✅ File zip berhasil dihapus: {zip_path}")
                except Exception as e:
                    errors.append(f"❌ Gagal menghapus {zip_path}: {str(e)}")
                    print(f"❌ Error: {str(e)}")
        except Exception as e:
            errors.append(f"❌ Gagal mendapatkan daftar file zip: {str(e)}")
            print(f"❌ Error: {str(e)}")
        
        # Bersihkan folder upload
        try:
            upload_files = os.listdir(UPLOAD_DIR)
            for file in upload_files:
                file_path = os.path.join(UPLOAD_DIR, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        files_deleted.append(file_path)
                        print(f"✅ File upload berhasil dihapus: {file_path}")
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        files_deleted.append(file_path)
                        print(f"✅ Direktori upload berhasil dihapus: {file_path}")
                except Exception as e:
                    errors.append(f"❌ Gagal menghapus {file_path}: {str(e)}")
                    print(f"❌ Error: {str(e)}")
        except Exception as e:
            errors.append(f"❌ Gagal mendapatkan daftar file di folder upload: {str(e)}")
            print(f"❌ Error: {str(e)}")
        
        # Return status dan detailnya
        return {
            "success": len(files_deleted) > 0,
            "files_deleted": files_deleted,
            "errors": errors
        }
    except Exception as e:
        print(f"❌ Error umum dalam cleanup: {str(e)}")
        return {
            "success": False,
            "files_deleted": files_deleted,
            "errors": [f"Error umum: {str(e)}"] + errors
        }

@app.route('/', methods=['GET', 'POST'])
def index():
    global df, stock_name, start_date, end_date  # Simpan informasi saham
    df = None
    stock_name = ''

    if request.method == 'POST':
        stock_name = request.form['stock_name']
        start_date = request.form['start_date']
        end_date = request.form['end_date']

        try:
            df = yf.download(stock_name, start=start_date, end=end_date)
            df.reset_index(inplace=True)
            # Tambahkan kolom "No" mulai dari 1
            df.insert(0, "No", range(1, len(df) + 1))
            # Format angka menjadi 2 digit di belakang koma
            df = df.round(2)
            # Hapus header tambahan "Price" jika muncul
            if isinstance(df.columns, pd.MultiIndex):  
                df.columns = df.columns.droplevel(1)
            # Hapus nama indeks agar tidak muncul ticker
            df.columns.name = None 
            # Konversi ke HTML dengan kelas Bootstrap
            df_html = df.to_html(classes="table table-bordered text-center", border=0, index=False)
            # Tambahkan class 'table-dark' ke <thead>
            df_html = df_html.replace('<thead>', '<thead class="table-dark text-center">')

        except Exception as e:
            return toast_redirect('index', f'Gagal mengambil data saham: {str(e)}. Periksa kode saham dan koneksi internet Anda.', 'error')
        return render_template('index.html', data=df_html, stock_name=stock_name, active_page='home')
    return render_template('index.html', data=None, active_page='home')

@app.route('/download_csv')
def download_csv():
    if df is not None:
        filename = f"{stock_name}_data.csv"
        filepath = os.path.join("static", filename)
        df.to_csv(filepath, index=False)
        return send_file(filepath, as_attachment=True)
    return toast_redirect('index', 'Data belum tersedia! Ambil data saham terlebih dahulu.', 'warning')

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    global df, train_data, test_data, scalers, time_step

    if request.method == 'POST':
        # Ambil parameter dari form
        epochs = int(request.form['epochs'])
        batch_size = int(request.form['batch_size'])
        neurons = int(request.form['neurons'])
        
        # Parameter baru
        time_step = int(request.form.get('time_step', 5))
        learning_rate = float(request.form.get('learning_rate', 0.001))
        dropout_rate = float(request.form.get('dropout_rate', 0.2))
        lstm_layers = int(request.form.get('lstm_layers', 2))
        
        # Simpan parameter untuk digunakan nanti
        model_params = {
            'epochs': epochs,
            'batch_size': batch_size,
            'neurons': neurons,
            'time_step': time_step,
            'learning_rate': learning_rate,
            'dropout_rate': dropout_rate,
            'lstm_layers': lstm_layers
        }
        
        # Simpan parameter ke file JSON
        params_path = os.path.join(DATA_DIR, "model_params.json")
        with open(params_path, 'w') as f:
            json.dump(model_params, f)

        if df is None or df.empty:
            return toast_redirect('index', 'Data saham belum tersedia! Kembali ke Beranda, pilih saham dan klik "Ambil Data Saham" terlebih dahulu.', 'warning')

        # Buat salinan data dan hapus kolom 'No'
        processed_df = df.copy()
        
        # Simpan kolom 'Date' untuk referensi
        date_column = None
        if 'Date' in processed_df.columns:
            date_column = processed_df['Date'].copy()
            processed_df = processed_df.drop(columns=['Date'])
        
        # Hapus kolom 'No'
        if 'No' in processed_df.columns:
            processed_df = processed_df.drop(columns=['No'])

        # Pastikan Volume termasuk dalam data
        if 'Volume' not in processed_df.columns:
            return toast_redirect('index', 'Kolom Volume tidak ditemukan dalam data. Coba pilih saham lain atau periksa koneksi internet.', 'warning')
            
        # Normalisasi data
        scalers = {}
        scaled_data = pd.DataFrame()
        
        for col in processed_df.columns:
            scaler = MinMaxScaler()
            scaled_data[col] = scaler.fit_transform(processed_df[col].values.reshape(-1, 1)).flatten()
            scalers[col] = scaler

        # Split data 80% training - 20% testing
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data.iloc[:train_size].values.astype(np.float32)
        test_data = scaled_data.iloc[train_size:].values.astype(np.float32)

        if len(test_data) == 0:
            return toast_redirect('index', 'Data testing kosong. Pilih rentang tanggal yang lebih panjang (minimal 2 bulan) lalu ambil data ulang.', 'warning')

        # Simpan dataframe ke JSON
        df_path = os.path.join(DATA_DIR, "df.json")
        df_to_save = df.copy()
        if 'Date' in df_to_save.columns:
            df_to_save['Date'] = df_to_save['Date'].astype(str)
        df_to_save.to_json(df_path)
        
        # Simpan scaler
        scaler_path = os.path.join(MODEL_DIR, "scalers.pkl")
        joblib.dump(scalers, scaler_path)
        
        # Simpan data training & testing
        train_data_path = os.path.join(DATA_DIR, "train_data.json")
        test_data_path = os.path.join(DATA_DIR, "test_data.json")

        with open(train_data_path, "w") as f:
            json.dump(train_data.tolist(), f)

        with open(test_data_path, "w") as f:
            json.dump(test_data.tolist(), f)

        # Buat DataFrame untuk tampilan
        columns = processed_df.columns.tolist()
        train_df = pd.DataFrame(train_data, columns=columns).round(4)
        test_df = pd.DataFrame(test_data, columns=columns).round(4)

        train_df.insert(0, "No", range(1, len(train_df) + 1))
        test_df.insert(0, "No", range(1, len(test_df) + 1))

        # Konversi ke HTML
        train_html = train_df.to_html(classes="table table-bordered text-center", border=0, index=False)
        test_html = test_df.to_html(classes="table table-bordered text-center", border=0, index=False)

        train_html = train_html.replace('<thead>', '<thead class="table-dark text-center">')
        test_html = test_html.replace('<thead>', '<thead class="table-dark text-center">')

        return render_template('train_result.html',
                              epochs=epochs, batch_size=batch_size, neurons=neurons,
                              time_step=time_step, learning_rate=learning_rate,
                              dropout_rate=dropout_rate, lstm_layers=lstm_layers,
                              train_df=train_html, 
                              test_df=test_html, 
                              train_count=len(train_df), 
                              test_count=len(test_df), 
                              next_step=url_for('train_lstm'))
    return render_template('train.html', active_page='train')

@app.route('/train_lstm', methods=['GET', 'POST'])
def train_lstm():
    if request.method == 'GET':
        return render_template('train_complete.html', active_page='train')

    # Cek apakah file data saham ada
    train_data_path = os.path.join(DATA_DIR, "train_data.json")
    test_data_path = os.path.join(DATA_DIR, "test_data.json")
    params_path = os.path.join(DATA_DIR, "model_params.json")
    
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        return toast_redirect('train_model', 'Data preprocessing belum tersedia. Kembali ke halaman Training, isi parameter, dan klik "Preprocess Data" terlebih dahulu.', 'warning')

    # Load training & testing data dari JSON
    with open(train_data_path, "r") as f:
        train_data = np.array(json.load(f), dtype=np.float32)

    with open(test_data_path, "r") as f:
        test_data = np.array(json.load(f), dtype=np.float32)
        
    # Load model parameters
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            model_params = json.load(f)
            epochs = model_params.get('epochs', 50)
            batch_size = model_params.get('batch_size', 32)
            neurons = model_params.get('neurons', 50)
            time_step = model_params.get('time_step', 5)
            learning_rate = model_params.get('learning_rate', 0.001)
            dropout_rate = model_params.get('dropout_rate', 0.2)
            lstm_layers = model_params.get('lstm_layers', 2)
    else:
        # Ambil parameter dari form jika tidak ada file parameter
        try:
            epochs = int(request.form['epochs'])
            batch_size = int(request.form['batch_size'])
            neurons = int(request.form['neurons'])
            time_step = int(request.form.get('time_step', 5))
            learning_rate = float(request.form.get('learning_rate', 0.001))
            dropout_rate = float(request.form.get('dropout_rate', 0.2))
            lstm_layers = int(request.form.get('lstm_layers', 2))
        except KeyError:
            return toast_redirect('train_model', 'Parameter training tidak lengkap. Pastikan semua field (Epochs, Batch Size, Neurons) terisi.', 'warning')

    # Membuat dataset untuk LSTM
    # Pastikan data cukup untuk training dan testing
    if len(train_data) <= time_step:
        return toast_redirect('index', 'Data tidak cukup untuk training (butuh lebih dari ' + str(time_step) + ' baris data). Pilih rentang waktu yang lebih panjang, lalu ambil data ulang.', 'warning')
    if len(test_data) <= time_step:
        return toast_redirect('index', 'Data testing tidak cukup. Pilih rentang waktu yang lebih panjang agar data testing memadai.', 'warning')

    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    # Import Early Stopping
    from tensorflow.keras.callbacks import EarlyStopping
    
    # ===========================
    # 📌 1. Definisi Model LSTM dengan parameter yang disesuaikan
    # ===========================
    model = Sequential()
    
    # Input Layer
    model.add(LSTM(neurons, return_sequences=(lstm_layers > 1), 
                   input_shape=(time_step, train_data.shape[1])))
    model.add(Dropout(dropout_rate))
    
    # Hidden Layers (Optional)
    if lstm_layers >= 3:
        model.add(LSTM(neurons*2, return_sequences=True))
        model.add(Dropout(dropout_rate))
    
    if lstm_layers >= 2:
        # Second Layer
        model.add(LSTM(neurons, return_sequences=False))
        model.add(Dropout(dropout_rate))
    
    # Output Layer
    model.add(Dense(train_data.shape[1]))

    model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=learning_rate))

    # Early stopping untuk mencegah overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,  # Hentikan jika tidak ada peningkatan setelah 10 epoch
        restore_best_weights=True
    )

    # ===========================
    # 📌 2. Training Model
    # ===========================
    history = model.fit(
        X_train, Y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, Y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # ===========================
    # 📌 3. Simpan Model dan Parameter
    # ===========================
    model_path = os.path.join(MODEL_DIR, "lstm_model.h5")
    model.save(model_path)
    
    # Simpan parameter model yang digunakan
    model_params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'neurons': neurons,
        'time_step': time_step,
        'learning_rate': learning_rate,
        'dropout_rate': dropout_rate,
        'lstm_layers': lstm_layers,
        'actual_epochs': len(history.history['loss'])  # Jumlah epoch yang sebenarnya dijalankan
    }
    
    params_path = os.path.join(MODEL_DIR, "model_params.json")
    with open(params_path, 'w') as f:
        json.dump(model_params, f)
    
    # ===========================
    # 📌 4. Simpan paketan model untuk download
    # ===========================
    # Buat folder zip untuk menyimpan semua file yang dibutuhkan
    model_package_dir = os.path.join(MODEL_DIR, "model_package")
    os.makedirs(model_package_dir, exist_ok=True)
    
    # Copy model dan scaler ke folder paket
    shutil.copy(model_path, os.path.join(model_package_dir, "lstm_model.h5"))
    shutil.copy(os.path.join(MODEL_DIR, "scalers.pkl"), os.path.join(model_package_dir, "scalers.pkl"))
    shutil.copy(params_path, os.path.join(model_package_dir, "model_params.json"))
    
    # Buat nama file yang mencakup nama saham, periode, dan parameter tuning
    model_filename = f"{stock_name}_{start_date}_to_{end_date}_ep{epochs}_bs{batch_size}_n{neurons}_ts{time_step}_lr{learning_rate}_dr{dropout_rate}_l{lstm_layers}"
    # Bersihkan karakter yang tidak valid untuk nama file
    model_filename = "".join(c if c.isalnum() or c in ['-', '_', '.'] else '_' for c in model_filename)
    
    # Buat zip file dari folder paket dengan nama yang menyesuaikan
    zip_path = os.path.join(MODEL_DIR, f"{model_filename}.zip")
    shutil.make_archive(os.path.join(MODEL_DIR, model_filename), 'zip', model_package_dir)

    # Visualisasi loss history
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Simpan grafik dalam bentuk base64 untuk ditampilkan di HTML
    img = io.BytesIO()
    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    loss_graph_url = base64.b64encode(img.getvalue()).decode()

    return render_template('train_complete.html',
                           active_page='train',
                           model_path=model_path,
                           zip_path=zip_path, 
                           model_filename=model_filename,
                           epochs=epochs,
                           actual_epochs=len(history.history['loss']),
                           batch_size=batch_size,
                           neurons=neurons,
                           time_step=time_step,
                           learning_rate=learning_rate,
                           dropout_rate=dropout_rate,
                           lstm_layers=lstm_layers,
                           loss_graph_url=loss_graph_url)

@app.route('/download_model')
def download_model():
    # Dapatkan nama file dari request
    model_filename = request.args.get('filename', 'stock_prediction_model')
    zip_path = os.path.join(MODEL_DIR, f"{model_filename}.zip")
    
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True, 
                         download_name=f"{model_filename}.zip")
    return toast_redirect('index', 'Model belum tersedia untuk diunduh!')

@app.route('/upload_model', methods=['POST'])
def upload_model():
    # Check if the post request has the file part
    if 'model_file' not in request.files:
        return toast_redirect('direct_predict', 'File model tidak ditemukan!')
    
    file = request.files['model_file']
    
    # If user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return toast_redirect('direct_predict', 'Tidak ada file yang dipilih!')
    
    if file and file.filename.endswith('.zip'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        
        # Extract zip file
        import zipfile
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(UPLOAD_DIR)
            
            # Copy extracted files to model directory
            if os.path.exists(os.path.join(UPLOAD_DIR, "lstm_model.h5")):
                shutil.copy(os.path.join(UPLOAD_DIR, "lstm_model.h5"), os.path.join(MODEL_DIR, "lstm_model.h5"))
            
            if os.path.exists(os.path.join(UPLOAD_DIR, "scalers.pkl")):
                shutil.copy(os.path.join(UPLOAD_DIR, "scalers.pkl"), os.path.join(MODEL_DIR, "scalers.pkl"))
                
            auto_predict = request.form.get('auto_predict') == 'true'
    
            if auto_predict:
                # Redirect ke halaman prediksi dengan JavaScript untuk auto-submit
                return render_template('auto_submit.html', 
                                    next_url=url_for('predict'))
            else:
                return redirect(url_for('predict'))
        except Exception as e:
            return toast_redirect('direct_predict', f'Error extracting model: {str(e)}')
    
    return toast_redirect('direct_predict', 'Format file tidak valid! Harap upload file .zip')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Check if we need to use an uploaded model
        model_source = request.form.get('model_source', 'current')

        # Add this line to get cleanup parameter
        cleanup_after = request.form.get('cleanup', 'false') == 'true'
        
        # Cek apakah file data tersedia
        df_path = os.path.join(DATA_DIR, "df.json")
        train_data_path = os.path.join(DATA_DIR, "train_data.json")
        test_data_path = os.path.join(DATA_DIR, "test_data.json")
        model_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        scaler_path = os.path.join(MODEL_DIR, "scalers.pkl")

        # Pastikan semua file yang diperlukan ada
        if not all(os.path.exists(path) for path in [df_path, train_data_path, test_data_path, model_path, scaler_path]):
            return toast_redirect('index', 'Data atau model belum tersedia. Mulai dari awal: ambil data saham → preprocessing → training → prediksi.', 'warning')

        # Load data saham dari JSON
        df = pd.read_json(df_path)

        if df.empty:
            return toast_redirect('index', 'Data tidak valid. Coba ambil ulang data saham dengan rentang waktu yang berbeda.', 'warning')

        # Load training & testing data dari JSON
        with open(test_data_path, "r") as f:
            test_data = np.array(json.load(f), dtype=np.float32)

        # Load model & scaler
        model = load_model(model_path)
        scalers = joblib.load(scaler_path)

        # Load model parameters
        params_path = os.path.join(DATA_DIR, "model_params.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                model_params = json.load(f)
                time_step = model_params.get('time_step', 5)  # Gunakan time_step dari training
        else:
            time_step = 5  # Fallback ke default jika tidak ada file parameter

        X_test, Y_test = create_dataset(test_data, time_step)

        # Extract dates for plotting
        test_dates = None
        if 'Date' in df.columns:
            # Get dates corresponding to test data, accounting for time_step offset
            train_size = int(len(df) * 0.8)
            all_test_dates = df['Date'].iloc[train_size:].reset_index(drop=True)
            # We need to skip the first `time_step` dates since they're used for prediction input
            test_dates = all_test_dates.iloc[time_step:].reset_index(drop=True)
            # Make sure we have the right number of dates
            if len(test_dates) > len(Y_test):
                test_dates = test_dates[:len(Y_test)]
            elif len(test_dates) < len(Y_test):
                # Handle case where we have more predictions than dates
                test_dates = pd.Series(range(len(Y_test)))
        else:
            # If no dates available, use indices
            test_dates = pd.Series(range(len(Y_test)))

        # Format date if it's string
        if test_dates is not None and isinstance(test_dates.iloc[0], str):
            test_dates = pd.to_datetime(test_dates)

        # Prediksi menggunakan model
        predictions = model.predict(X_test)
        
        # Pastikan Y_test ada data
        if Y_test.size == 0:
            return toast_redirect('predict', 'Tidak ada data pengujian yang tersedia untuk evaluasi.')

        # Ambil nama kolom dari scalers
        columns = list(scalers.keys())

        # Denormalisasi hasil prediksi dan data aktual
        predictions_denorm = np.zeros_like(predictions)
        y_test_denorm = np.zeros_like(Y_test)

        for i, col in enumerate(columns):
            if i < predictions.shape[1]:  # Pastikan indeks tidak melebihi dimensi
                predictions_denorm[:, i] = scalers[col].inverse_transform(predictions[:, i].reshape(-1, 1)).flatten()
                y_test_denorm[:, i] = scalers[col].inverse_transform(Y_test[:, i].reshape(-1, 1)).flatten()

        # Buat DataFrame dengan hasil
        result_df = pd.DataFrame()
        
        # Tambahkan data aktual dan prediksi untuk setiap kolom
        for i, col in enumerate(columns):
            if i < predictions.shape[1]:  # Pastikan indeks tidak melebihi dimensi
                result_df[f'Actual_{col}'] = y_test_denorm[:, i]
                result_df[f'Predicted_{col}'] = predictions_denorm[:, i]

        # Tambahkan kolom tanggal jika tersedia
        if test_dates is not None:
            # Jika test_dates adalah pandas Series dengan datetime objects
            if isinstance(test_dates.iloc[0], pd.Timestamp):
                # Format tanggal menjadi string untuk tampilan lebih baik
                formatted_dates = test_dates.dt.strftime('%d-%m-%Y')
                result_df.insert(0, "Tanggal", formatted_dates)
            else:
                # Jika bukan datetime, gunakan apa adanya
                result_df.insert(0, "Tanggal", test_dates)

        # Tambahkan kolom nomor
        result_df.insert(0, "No", range(1, len(result_df) + 1))
        
        # Format angka menjadi 2 digit di belakang koma
        result_df = result_df.round(2)
        
        # Konversi ke HTML dengan kelas Bootstrap seperti tabel preprocessing
        predictions_html = result_df.to_html(classes="table table-bordered text-center", border=0, index=False)
        # Tambahkan class 'table-dark' ke <thead>
        predictions_html = predictions_html.replace('<thead>', '<thead class="table-dark text-center">')

        # Simpan hasil prediksi ke CSV
        predictions_csv = "static/predictions.csv"
        result_df.to_csv(predictions_csv, index=False)

        # Plot hasil prediksi vs aktual untuk Close dan Volume
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(14, 25)) 

        # Format dates on x-axis if we have datetime objects
        date_formatter = plt.matplotlib.dates.DateFormatter('%d-%m-%Y')

        # Plot untuk Close
        close_index = columns.index('Close') if 'Close' in columns else 3
        ax1.plot(test_dates, y_test_denorm[:, close_index], label="Actual Close", color='blue')
        ax1.plot(test_dates, predictions_denorm[:, close_index], label="Predicted Close", color='red', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax1.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.legend()
        ax1.set_title("Perbandingan Harga Saham (Close) Aktual vs Prediksi")
        ax1.set_xlabel("Tanggal")
        ax1.set_ylabel("Harga Saham")

        # Plot untuk Volume
        volume_index = columns.index('Volume') if 'Volume' in columns else 5
        ax2.plot(test_dates, y_test_denorm[:, volume_index], label="Actual Volume", color='green')
        ax2.plot(test_dates, predictions_denorm[:, volume_index], label="Predicted Volume", color='orange', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax2.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.legend()
        ax2.set_title("Perbandingan Volume Saham Aktual vs Prediksi")
        ax2.set_xlabel("Tanggal")
        ax2.set_ylabel("Volume")

        # Plot untuk High
        high_index = columns.index('High') if 'High' in columns else 2
        ax3.plot(test_dates, y_test_denorm[:, high_index], label="Actual High", color='green')
        ax3.plot(test_dates, predictions_denorm[:, high_index], label="Predicted High", color='orange', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax3.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        ax3.legend()
        ax3.set_title("Perbandingan High Saham Aktual vs Prediksi")
        ax3.set_xlabel("Tanggal")
        ax3.set_ylabel("High")

        # Plot untuk Open
        open_index = columns.index('Open') if 'Open' in columns else 1
        ax4.plot(test_dates, y_test_denorm[:, open_index], label="Actual Open", color='green')
        ax4.plot(test_dates, predictions_denorm[:, open_index], label="Predicted Open", color='orange', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax4.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        ax4.legend()
        ax4.set_title("Perbandingan Open Saham Aktual vs Prediksi")
        ax4.set_xlabel("Tanggal")
        ax4.set_ylabel("Open")

        # Plot untuk Low
        low_index = columns.index('Low') if 'Low' in columns else 4
        ax5.plot(test_dates, y_test_denorm[:, low_index], label="Actual Low", color='green')
        ax5.plot(test_dates, predictions_denorm[:, low_index], label="Predicted Low", color='orange', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax5.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        ax5.legend()
        ax5.set_title("Perbandingan Low Saham Aktual vs Prediksi")
        ax5.set_xlabel("Tanggal")
        ax5.set_ylabel("Low")

        # Simpan grafik dalam bentuk base64 untuk ditampilkan di HTML
        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        # Hitung error untuk Close dan Volume
        close_errors = calculate_errors(y_test_denorm[:, close_index], predictions_denorm[:, close_index])
        volume_errors = calculate_errors(y_test_denorm[:, volume_index], predictions_denorm[:, volume_index])
        high_errors = calculate_errors(y_test_denorm[:, high_index], predictions_denorm[:, high_index])
        open_errors = calculate_errors(y_test_denorm[:, open_index], predictions_denorm[:, open_index])
        low_errors = calculate_errors(y_test_denorm[:, low_index], predictions_denorm[:, low_index])
        
        error_analysis = {
            "Close": close_errors,
            "Volume": volume_errors,
            "High": high_errors,
            "Open": open_errors,
            "Low": low_errors
        }
        
        # Analisis tren harga saham
        trend_message, trend = analyze_price_trend(y_test_denorm[:, close_index], predictions_denorm[:, close_index])

        # Siapkan data chart interaktif (JSON)
        chart_dates = []
        if test_dates is not None:
            if isinstance(test_dates.iloc[0], pd.Timestamp):
                chart_dates = test_dates.dt.strftime('%d-%m-%Y').tolist()
            else:
                chart_dates = [str(d) for d in test_dates.tolist()]
        else:
            chart_dates = [str(i) for i in range(len(y_test_denorm))]

        chart_data = {
            'dates': chart_dates,
            'series': {}
        }
        for col_name, col_idx in [('Close', close_index), ('High', high_index), ('Low', low_index), ('Open', open_index), ('Volume', volume_index)]:
            if col_idx < predictions_denorm.shape[1]:
                chart_data['series'][col_name] = {
                    'actual': [round(v, 2) for v in y_test_denorm[:, col_idx].tolist()],
                    'predicted': [round(v, 2) for v in predictions_denorm[:, col_idx].tolist()]
                }

        # Di bagian akhir fungsi predict
        if cleanup_after:
            cleanup_result = cleanup_model_files()
            cleanup_message = f"✅ Berhasil menghapus {len(cleanup_result['files_deleted'])} file." if cleanup_result["success"] else "❌ Gagal menghapus beberapa file."
        else:
            cleanup_message = None

        return render_template('prediction.html',
                              active_page='predict',
                              stock_name=stock_name,
                              timeframe='Harian',
                              predictions_html=predictions_html,
                              predictions_csv=predictions_csv,
                              graph_url=graph_url,
                              chart_data=chart_data,
                              error_analysis=error_analysis,
                              trend_message=trend_message,
                              trend=trend,
                              cleanup_message=cleanup_message)
    # Guard: cek apakah model sudah tersedia
    model_path = os.path.join(MODEL_DIR, "lstm_model.h5")
    scaler_path = os.path.join(MODEL_DIR, "scalers.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return toast_redirect('train_model', 'Model belum tersedia. Lakukan training model terlebih dahulu sebelum prediksi.', 'warning')
    return render_template('predict_options.html', active_page='predict')

@app.route('/cleanup', methods=['GET'])
def cleanup():
    result = cleanup_model_files()
    
    if result["success"]:
        msg = f'Berhasil membersihkan {len(result["files_deleted"])} file.'
        if result["errors"]:
            msg += f' ({len(result["errors"])} error)'
        toast_type = 'success'
    else:
        msg = 'Gagal menghapus beberapa file.'
        toast_type = 'error'
    
    # Print detail hasil untuk debugging
    print("\n===== HASIL CLEANUP =====")
    print(f"Files deleted: {len(result['files_deleted'])}")
    print(f"Errors: {len(result['errors'])}")
    if result["errors"]:
        print("\nDetail errors:")
        for err in result["errors"]:
            print(f"- {err}")
    print("========================\n")
    
    return toast_redirect('index', msg, toast_type)

@app.route('/direct_predict', methods=['GET', 'POST'])
def direct_predict():
    if request.method == 'GET':
        return render_template('direct_predict.html', active_page='predict')
    
    # Handle POST request - upload model and predict
    try:
        if 'model_file' not in request.files:
            return toast_redirect('direct_predict', 'File model tidak ditemukan!')
        
        file = request.files['model_file']
        if file.filename == '' or not file.filename.endswith('.zip'):
            return toast_redirect('direct_predict', 'File model tidak valid! Harap upload file .zip')
        
        # Save and extract model
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        
        import zipfile
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(UPLOAD_DIR)
        
        # Copy extracted files to model directory
        required_files = ['lstm_model.h5', 'scalers.pkl', 'model_params.json']
        for req_file in required_files:
            src_path = os.path.join(UPLOAD_DIR, req_file)
            dst_path = os.path.join(MODEL_DIR, req_file)
            if os.path.exists(src_path):
                shutil.copy(src_path, dst_path)
            else:
                return toast_redirect('direct_predict', f'File {req_file} tidak ditemukan dalam model!')
        
        # Get form data and download stock data
        stock_name = request.form['stock_name']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        prediction_timeframe = request.form.get('prediction_timeframe', 'daily')
        cleanup_after = request.form.get('cleanup', 'false') == 'true'
        
        interval = '1d'
        if prediction_timeframe == 'monthly':
            interval = '1mo'
        elif prediction_timeframe == 'weekly':
            interval = '1wk'
        
        df = yf.download(stock_name, start=start_date, end=end_date, interval=interval)
        if df.empty:
            return toast_redirect('direct_predict', 'Tidak dapat mengambil data saham. Periksa kode saham dan koneksi internet!')
        
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns.name = None
        
        # Save df for prediction
        df_path = os.path.join(DATA_DIR, "df.json")
        df_to_save = df.copy()
        if 'Date' in df_to_save.columns:
            df_to_save['Date'] = df_to_save['Date'].astype(str)
        df_to_save.to_json(df_path)
        
        # Preprocess data
        processed_df = df.copy()
        if 'Date' in processed_df.columns:
            processed_df = processed_df.drop(columns=['Date'])
        
        scalers = {}
        scaled_data = pd.DataFrame()
        for col in processed_df.columns:
            scaler = MinMaxScaler()
            scaled_data[col] = scaler.fit_transform(processed_df[col].values.reshape(-1, 1)).flatten()
            scalers[col] = scaler
        
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data.iloc[:train_size].values.astype(np.float32)
        test_data = scaled_data.iloc[train_size:].values.astype(np.float32)
        
        # Save scaler and data
        joblib.dump(scalers, os.path.join(MODEL_DIR, "scalers.pkl"))
        
        with open(os.path.join(DATA_DIR, "train_data.json"), "w") as f:
            json.dump(train_data.tolist(), f)
        with open(os.path.join(DATA_DIR, "test_data.json"), "w") as f:
            json.dump(test_data.tolist(), f)
        
        # Load model params
        params_path = os.path.join(MODEL_DIR, "model_params.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                model_params = json.load(f)
            time_step = model_params.get('time_step', 5)
        else:
            time_step = 5
        
        # Run prediction
        model = load_model(os.path.join(MODEL_DIR, "lstm_model.h5"))
        X_test, Y_test = create_dataset(test_data, time_step)
        
        test_dates = None
        if 'Date' in df.columns:
            all_test_dates = df['Date'].iloc[train_size:].reset_index(drop=True)
            test_dates = all_test_dates.iloc[time_step:].reset_index(drop=True)
            if len(test_dates) > len(Y_test):
                test_dates = test_dates[:len(Y_test)]
            elif len(test_dates) < len(Y_test):
                test_dates = pd.Series(range(len(Y_test)))
        else:
            test_dates = pd.Series(range(len(Y_test)))
        
        if test_dates is not None and isinstance(test_dates.iloc[0], str):
            test_dates = pd.to_datetime(test_dates)
        
        predictions = model.predict(X_test)
        
        if Y_test.size == 0:
            return toast_redirect('predict', 'Tidak ada data pengujian yang tersedia untuk evaluasi.')
        
        columns = list(scalers.keys())
        predictions_denorm = np.zeros_like(predictions)
        y_test_denorm = np.zeros_like(Y_test)
        
        for i, col in enumerate(columns):
            if i < predictions.shape[1]:
                predictions_denorm[:, i] = scalers[col].inverse_transform(predictions[:, i].reshape(-1, 1)).flatten()
                y_test_denorm[:, i] = scalers[col].inverse_transform(Y_test[:, i].reshape(-1, 1)).flatten()
        
        result_df = pd.DataFrame()
        for i, col in enumerate(columns):
            if i < predictions.shape[1]:
                result_df[f'Actual_{col}'] = y_test_denorm[:, i]
                result_df[f'Predicted_{col}'] = predictions_denorm[:, i]
        
        if test_dates is not None:
            if isinstance(test_dates.iloc[0], pd.Timestamp):
                formatted_dates = test_dates.dt.strftime('%d-%m-%Y')
                result_df.insert(0, "Tanggal", formatted_dates)
            else:
                result_df.insert(0, "Tanggal", test_dates)
        
        result_df.insert(0, "No", range(1, len(result_df) + 1))
        result_df = result_df.round(2)
        
        predictions_html = result_df.to_html(classes="table table-bordered text-center", border=0, index=False)
        predictions_html = predictions_html.replace('<thead>', '<thead class="table-dark text-center">')
        
        predictions_csv = "static/predictions.csv"
        result_df.to_csv(predictions_csv, index=False)
        
        # Plot
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(14, 25))
        date_formatter = plt.matplotlib.dates.DateFormatter('%d-%m-%Y')
        
        close_index = columns.index('Close') if 'Close' in columns else 3
        ax1.plot(test_dates, y_test_denorm[:, close_index], label="Actual Close", color='blue')
        ax1.plot(test_dates, predictions_denorm[:, close_index], label="Predicted Close", color='red', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax1.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.legend(); ax1.set_title("Close - Aktual vs Prediksi"); ax1.set_xlabel("Tanggal"); ax1.set_ylabel("Harga")
        
        volume_index = columns.index('Volume') if 'Volume' in columns else 5
        ax2.plot(test_dates, y_test_denorm[:, volume_index], label="Actual Volume", color='green')
        ax2.plot(test_dates, predictions_denorm[:, volume_index], label="Predicted Volume", color='orange', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax2.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        ax2.legend(); ax2.set_title("Volume - Aktual vs Prediksi"); ax2.set_xlabel("Tanggal"); ax2.set_ylabel("Volume")
        
        high_index = columns.index('High') if 'High' in columns else 2
        ax3.plot(test_dates, y_test_denorm[:, high_index], label="Actual High", color='green')
        ax3.plot(test_dates, predictions_denorm[:, high_index], label="Predicted High", color='orange', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax3.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        ax3.legend(); ax3.set_title("High - Aktual vs Prediksi"); ax3.set_xlabel("Tanggal"); ax3.set_ylabel("High")
        
        open_index = columns.index('Open') if 'Open' in columns else 1
        ax4.plot(test_dates, y_test_denorm[:, open_index], label="Actual Open", color='green')
        ax4.plot(test_dates, predictions_denorm[:, open_index], label="Predicted Open", color='orange', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax4.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        ax4.legend(); ax4.set_title("Open - Aktual vs Prediksi"); ax4.set_xlabel("Tanggal"); ax4.set_ylabel("Open")
        
        low_index = columns.index('Low') if 'Low' in columns else 4
        ax5.plot(test_dates, y_test_denorm[:, low_index], label="Actual Low", color='green')
        ax5.plot(test_dates, predictions_denorm[:, low_index], label="Predicted Low", color='orange', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax5.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        ax5.legend(); ax5.set_title("Low - Aktual vs Prediksi"); ax5.set_xlabel("Tanggal"); ax5.set_ylabel("Low")
        
        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        
        # Error metrics
        close_errors = calculate_errors(y_test_denorm[:, close_index], predictions_denorm[:, close_index])
        volume_errors = calculate_errors(y_test_denorm[:, volume_index], predictions_denorm[:, volume_index])
        high_errors = calculate_errors(y_test_denorm[:, high_index], predictions_denorm[:, high_index])
        open_errors = calculate_errors(y_test_denorm[:, open_index], predictions_denorm[:, open_index])
        low_errors = calculate_errors(y_test_denorm[:, low_index], predictions_denorm[:, low_index])
        
        error_analysis = {
            "Close": close_errors, "Volume": volume_errors,
            "High": high_errors, "Open": open_errors, "Low": low_errors
        }
        
        trend_message, trend = analyze_price_trend(y_test_denorm[:, close_index], predictions_denorm[:, close_index])

        # Siapkan data chart interaktif (JSON)
        chart_dates = []
        if test_dates is not None:
            if isinstance(test_dates.iloc[0], pd.Timestamp):
                chart_dates = test_dates.dt.strftime('%d-%m-%Y').tolist()
            else:
                chart_dates = [str(d) for d in test_dates.tolist()]
        else:
            chart_dates = [str(i) for i in range(len(y_test_denorm))]

        chart_data = {
            'dates': chart_dates,
            'series': {}
        }
        for col_name, col_idx in [('Close', close_index), ('High', high_index), ('Low', low_index), ('Open', open_index), ('Volume', volume_index)]:
            if col_idx < predictions_denorm.shape[1]:
                chart_data['series'][col_name] = {
                    'actual': [round(v, 2) for v in y_test_denorm[:, col_idx].tolist()],
                    'predicted': [round(v, 2) for v in predictions_denorm[:, col_idx].tolist()]
                }

        if cleanup_after:
            cleanup_result = cleanup_model_files()
            cleanup_message = f"✅ Berhasil menghapus {len(cleanup_result['files_deleted'])} file." if cleanup_result["success"] else "❌ Gagal menghapus beberapa file."
        else:
            cleanup_message = None

        return render_template('prediction.html',
                              active_page='predict',
                              stock_name=stock_name,
                              timeframe=prediction_timeframe,
                              predictions_html=predictions_html,
                              predictions_csv=predictions_csv,
                              graph_url=graph_url,
                              chart_data=chart_data,
                              error_analysis=error_analysis,
                              trend_message=trend_message,
                              trend=trend,
                              cleanup_message=cleanup_message)
    except Exception as e:
        return toast_redirect('direct_predict', f'Error: {str(e)}')

# ===== REALTIME STOCK DATA WITH SELENIUM =====
@app.route('/realtime', methods=['GET', 'POST'])
def realtime():
    if request.method == 'GET':
        return render_template('realtime.html', active_page='realtime')
    
    try:
        stock_code = request.form.get('stock_name', '').strip()
        if not stock_code:
            return toast_redirect('realtime', 'Pilih kode saham terlebih dahulu!', 'warning')
        
        # Scrape real-time data using Selenium
        data = scrape_yahoo_finance(stock_code)
        
        if data is None:
            return toast_redirect('realtime', f'Gagal mengambil data {stock_code} dari Yahoo Finance. Periksa kode saham atau coba lagi.', 'error')
        
        # Also get recent historical data for mini chart (last 30 days)
        try:
            recent_df = yf.download(stock_code, period='1mo', interval='1d', progress=False)
            recent_data = []
            if not recent_df.empty:
                recent_df.reset_index(inplace=True)
                if isinstance(recent_df.columns, pd.MultiIndex):
                    recent_df.columns = recent_df.columns.droplevel(1)
                recent_df.columns.name = None
                for _, row in recent_df.iterrows():
                    date_str = str(row['Date'])[:10] if 'Date' in recent_df.columns else str(row.iloc[0])[:10]
                    close_val = float(row['Close']) if 'Close' in recent_df.columns else float(row.iloc[4])
                    recent_data.append({'date': date_str, 'close': round(close_val, 2)})
        except Exception:
            recent_data = []
        
        # Check if model exists for prediction
        model_exists = os.path.exists(os.path.join(MODEL_DIR, 'lstm_model.h5'))
        prediction = None
        
        if model_exists:
            try:
                scalers = joblib.load(os.path.join(MODEL_DIR, 'scalers.pkl'))
                model = load_model(os.path.join(MODEL_DIR, 'lstm_model.h5'))
                
                with open(os.path.join(MODEL_DIR, 'model_params.json'), 'r') as f:
                    model_params = json.load(f)
                time_step = model_params.get('time_step', 5)
                
                # Get recent data for prediction window
                hist_df = yf.download(stock_code, period='3mo', interval='1d', progress=False)
                if not hist_df.empty:
                    hist_df.reset_index(inplace=True)
                    if isinstance(hist_df.columns, pd.MultiIndex):
                        hist_df.columns = hist_df.columns.droplevel(1)
                    hist_df.columns.name = None
                    
                    if 'Date' in hist_df.columns:
                        hist_df = hist_df.drop(columns=['Date'])
                    
                    # Scale the data
                    scaled_data = pd.DataFrame()
                    for col in hist_df.columns:
                        if col in scalers:
                            scaled_data[col] = scalers[col].transform(hist_df[col].values.reshape(-1, 1)).flatten()
                        else:
                            scaled_data[col] = MinMaxScaler().fit_transform(hist_df[col].values.reshape(-1, 1)).flatten()
                    
                    # Create dataset with last time_step rows
                    recent_scaled = scaled_data.iloc[-time_step:].values.astype(np.float32)
                    X_input = recent_scaled.reshape(1, time_step, recent_scaled.shape[1])
                    
                    # Predict
                    pred = model.predict(X_input, verbose=0)
                    
                    # Denormalize Close price prediction
                    close_idx = list(scalers.keys()).index('Close') if 'Close' in scalers else 3
                    if close_idx < pred.shape[1]:
                        pred_close = scalers['Close'].inverse_transform(pred[0, close_idx].reshape(-1, 1))[0][0]
                        prediction = round(float(pred_close), 2)
            except Exception as e:
                prediction = None
        
        return render_template('realtime.html',
                              active_page='realtime',
                              data=data,
                              recent_data=recent_data,
                              prediction=prediction,
                              model_exists=model_exists,
                              stock_code=stock_code)
    
    except Exception as e:
        return toast_redirect('realtime', f'Error: {str(e)}', 'error')


@app.route('/realtime_api/<stock_code>')
def realtime_api(stock_code):
    """API endpoint for AJAX real-time data refresh."""
    try:
        data = scrape_yahoo_finance(stock_code)
        if data:
            return json.dumps({'success': True, 'data': data}), 200, {'Content-Type': 'application/json'}
        return json.dumps({'success': False, 'error': 'Gagal mengambil data'}), 200, {'Content-Type': 'application/json'}
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)}), 500, {'Content-Type': 'application/json'}


if __name__ == '__main__':
    app.run(debug=True, exclude_patterns=['uploads/*', 'data/*', 'models/*', 'static/*'])