from flask import Flask, render_template, request, send_file, url_for, redirect
import pandas as pd
import yfinance as yf
import os
import io
import zipfile
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import base64
import joblib
import json
import shutil
from werkzeug.utils import secure_filename

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
direct_prediction_mode = False

# Fungsi untuk membuat dataset dalam bentuk sequence
def create_dataset(data, time_step):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i+time_step])
        Y.append(data[i+time_step])
    return np.array(X), np.array(Y)

def calculate_errors(actual, predicted, feature_name=None):
    # For volume data, scale down values before calculating errors
    scaling_factor = 1.0
    if feature_name == 'Volume':
        # Find a suitable scaling factor (e.g. 1 million)
        max_val = max(max(actual), max(predicted))
        if max_val > 1000000:  # If values are in millions or more
            scaling_factor = 1000000  # Scale by a million
            actual = actual / scaling_factor
            predicted = predicted / scaling_factor
    
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    
    # Calculate average absolute actual value for relative error
    avg_actual = np.mean(np.abs(actual))
    
    # Calculate relative errors as percentages
    mse_rel = (mse / avg_actual) if avg_actual != 0 else 0
    rmse_rel = (rmse / avg_actual * 100) if avg_actual != 0 else 0
    mae_rel = (mae / avg_actual * 100) if avg_actual != 0 else 0
    
    actual_trend = np.where(np.diff(np.append([actual[0]], actual)) > 0, 1, 0)
    pred_trend = np.where(np.diff(np.append([predicted[0]], predicted)) > 0, 1, 0)

    min_len = min(len(actual_trend), len(pred_trend))
    f1 = f1_score(actual_trend[:min_len], pred_trend[:min_len], average='binary')
    
    # Add info about scaling if it was applied
    scale_info = f" (scaled by {scaling_factor:,.0f})" if scaling_factor != 1.0 else ""
    
    return {
        "MSE": round(mse, 4),
        "MSE_rel": round(mse_rel, 2),
        "RMSE": round(rmse, 4),
        "RMSE_rel": round(rmse_rel, 2),  # RMSE as % of average value
        "MAE": round(mae, 4),
        "MAE_rel": round(mae_rel, 2),  # MAE as % of average value
        "F1": round(f1, 4),
        "scale_info": scale_info
    }

# Fungsi baru untuk menganalisis tren harga saham
def analyze_price_trend(actual_close, predicted_close, timeframe='daily'):
    # Sesuaikan jumlah periode terakhir berdasarkan timeframe
    if timeframe == 'daily':
        last_periods = 5  # 5 hari terakhir
        period_name = "hari"
    elif timeframe == 'weekly':
        last_periods = 4  # 4 minggu terakhir
        period_name = "minggu"
    elif timeframe == 'monthly':
        last_periods = 3  # 3 bulan terakhir
        period_name = "bulan"
    else:
        last_periods = 5  # Default
        period_name = "periode"
        
    # Pastikan kita memiliki cukup data
    if len(predicted_close) < last_periods + 1:
        last_periods = len(predicted_close) - 1
        if last_periods < 1:  # Jika data terlalu sedikit
            last_periods = 1
            
    # Analisis tren pada data prediksi sendiri
    first_pred = predicted_close[-last_periods]
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
    
    # Buat pesan analisis tren yang menyesuaikan dengan timeframe
    trend_message = f"Berdasarkan analisis {last_periods} {period_name} terakhir, harga saham diperkirakan akan {trend} dari {formatted_first} ke {formatted_last} (perubahan {percentage:.2f}%)."
    
    return trend_message, trend

# 5. You'll need to add a global variable at the top of the script
# Add this after the other global variable declarations
prediction_timeframe = 'daily'

def get_timeframe_display(timeframe):
    if timeframe == 'daily':
        return "Harian"
    elif timeframe == 'monthly':
        return "Bulanan"
    elif timeframe == 'weekly':
        return "Mingguan"
    return "Harian"

def cleanup_model_files():
    """Fungsi untuk menghapus file model dan data terkait setelah digunakan"""
    import gc  # Import garbage collector untuk pembersihan memori lebih efisien
    
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
        
        # Bersihkan memori yang tidak terpakai
        gc.collect()
        
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
    global df, stock_name, start_date, end_date, prediction_timeframe  # Add prediction_timeframe
    df = None
    stock_name = ''

    if request.method == 'POST':
        stock_name = request.form['stock_name']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        prediction_timeframe = request.form.get('prediction_timeframe', 'daily')  # Default to daily

        try:
            # Adjust interval based on prediction timeframe
            interval = '1d'  # Default daily
            if prediction_timeframe == 'monthly':
                interval = '1mo'
            elif prediction_timeframe == 'weekly':
                interval = '1wk'
                
            df = yf.download(stock_name, start=start_date, end=end_date, interval=interval)
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
            data = f"Error fetching data: {str(e)}"
        return render_template('index.html', data=df_html, stock_name=stock_name, prediction_timeframe=prediction_timeframe)
    return render_template('index.html', data=None)

@app.route('/download_csv')
def download_csv():
    if df is not None:
        filename = f"{stock_name}_data.csv"
        filepath = os.path.join("static", filename)
        df.to_csv(filepath, index=False)
        return send_file(filepath, as_attachment=True)
    return "Data belum tersedia!", 400

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    global df, train_data, test_data, scalers, time_step

    if request.method == 'POST':
        # Ambil parameter dari form
        epochs = int(request.form['epochs'])
        batch_size = int(request.form['batch_size'])
        neurons = int(request.form['neurons'])
        
        # Parameter baru
        # Sesuaikan default time_step berdasarkan timeframe
        if prediction_timeframe == 'daily':
            default_time_step = 5  # 5 hari untuk data harian
        elif prediction_timeframe == 'weekly':
            default_time_step = 4  # 4 minggu untuk data mingguan
        elif prediction_timeframe == 'monthly':
            default_time_step = 3  # 3 bulan untuk data bulanan
        else:
            default_time_step = 5
        
        time_step = int(request.form.get('time_step', default_time_step))
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
            return "❌ Data belum tersedia. Silakan ambil data saham terlebih dahulu! Atau mungkin anda lupa untuk online", 400

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
            return "❌ Data Volume tidak ditemukan dalam dataset!", 400
            
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
            return "❌ Data testing kosong. Pilih rentang waktu yang lebih panjang.", 400

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
    return render_template('train.html')

@app.route('/train_lstm', methods=['GET', 'POST'])
def train_lstm():
    if request.method == 'GET':
        return render_template('train_complete.html')

    # Cek apakah file data saham ada
    train_data_path = os.path.join(DATA_DIR, "train_data.json")
    test_data_path = os.path.join(DATA_DIR, "test_data.json")
    params_path = os.path.join(DATA_DIR, "model_params.json")
    
    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        return "❌ Data training belum tersedia. Lakukan persiapan data terlebih dahulu!", 400

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
            return "❌ Parameter tidak lengkap! Pastikan form terisi dengan benar.", 400

    # Membuat dataset untuk LSTM
    # Pastikan data cukup untuk training dan testing
    if len(train_data) <= time_step:
        return "❌ Data tidak cukup untuk training. Gunakan rentang waktu yang lebih panjang!", 400
    if len(test_data) <= time_step:
        return "❌ Data tidak cukup untuk testing. Gunakan rentang waktu yang lebih panjang!", 400

    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    
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
        patience=25,  # Hentikan jika tidak ada peningkatan setelah 15 epoch
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Kurangi learning rate setengahnya
        patience=10,  # Setelah 5 epoch tanpa perbaikan
        min_lr=0.00001,  # Minimal learning rate
        verbose=1
    )

    # Tambahkan ModelCheckpoint untuk menyimpan model terbaik selama training
    checkpoint_path = os.path.join(MODEL_DIR, "best_model_checkpoint.h5")
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )

    # ===========================
    # 📌 2. Training Model
    # ===========================
    history = model.fit(
        X_train, Y_train, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_test, Y_test),
        callbacks=[early_stopping, reduce_lr, checkpoint],
        verbose=1,
        shuffle=True  # Memastikan data di-shuffle setiap epoch
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
    
    # Buat nama file yang mencakup nama saham dan periode
    model_filename = f"{stock_name}_{start_date}_to_{end_date}"
    # Bersihkan karakter yang tidak valid untuk nama file
    model_filename = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in model_filename)
    
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
    return "Model belum tersedia untuk diunduh!", 400

@app.route('/upload_model', methods=['POST'])
def upload_model():
    # Check if the post request has the file part
    if 'model_file' not in request.files:
        return "❌ File model tidak ditemukan!", 400
    
    file = request.files['model_file']
    
    # If user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        return "❌ Tidak ada file yang dipilih!", 400
    
    if file and file.filename.endswith('.zip'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        
        # Extract zip file
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
            return f"❌ Error extracting model: {str(e)}", 400
    
    return "❌ Format file tidak valid! Harap upload file .zip", 400

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
            return "❌ Data atau model belum tersedia. Lakukan training terlebih dahulu!", 400

        # Load data saham dari JSON
        df = pd.read_json(df_path)

        if df.empty:
            return "❌ Data tidak valid. Pastikan data telah diambil dengan benar.", 400

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
            return "❌ Tidak ada data pengujian yang tersedia untuk evaluasi.", 400

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
                result_df[f'{col}'] = y_test_denorm[:, i]
                result_df[f'Pred_{col}'] = predictions_denorm[:, i]

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

        # Get the prediction timeframe from the stored global variable
        global prediction_timeframe
        timeframe_display = get_timeframe_display(prediction_timeframe)

        # Plot hasil prediksi vs aktual untuk Close dan Volume
        fig, (ax1, ax3, ax4, ax5) = plt.subplots(4, 1, figsize=(14, 25)) 
        
        # Sesuaikan format tanggal pada visualisasi berdasarkan timeframe
        if prediction_timeframe == 'daily':
            date_formatter = plt.matplotlib.dates.DateFormatter('%d-%m-%Y')
        elif prediction_timeframe == 'weekly':
            date_formatter = plt.matplotlib.dates.DateFormatter('%d-%m-%Y')  # Format sama, tapi interval berbeda
        elif prediction_timeframe == 'monthly':
            date_formatter = plt.matplotlib.dates.DateFormatter('%b %Y')  # Format bulan-tahun untuk data bulanan
        else:
            date_formatter = plt.matplotlib.dates.DateFormatter('%d-%m-%Y')
            
        # Sesuaikan interval tick pada axis x
        if prediction_timeframe == 'daily':
            ax1.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=max(1, len(test_dates)//10)))
        elif prediction_timeframe == 'weekly':
            ax1.xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator(interval=max(1, len(test_dates)//8)))
        elif prediction_timeframe == 'monthly':
            ax1.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=max(1, len(test_dates)//6)))

        # Plot untuk Close
        close_index = columns.index('Close') if 'Close' in columns else 3
        ax1.plot(test_dates, y_test_denorm[:, close_index], label="Actual Close", color='blue')
        ax1.plot(test_dates, predictions_denorm[:, close_index], label="Predicted Close", color='red', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax1.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.legend()
        ax1.set_title(f"Perbandingan Harga Saham (Close) Aktual vs Prediksi {timeframe_display}")
        ax1.set_xlabel("Tanggal")
        ax1.set_ylabel("Harga Saham")

        # Plot untuk Volume
        # volume_index = columns.index('Volume') if 'Volume' in columns else 5
        # ax2.plot(test_dates, y_test_denorm[:, volume_index], label="Actual Volume", color='green')
        # ax2.plot(test_dates, predictions_denorm[:, volume_index], label="Predicted Volume", color='orange', linestyle='dashed')
        # if isinstance(test_dates.iloc[0], pd.Timestamp):
        #     ax2.xaxis.set_major_formatter(date_formatter)
        #     plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        # ax2.legend()
        # ax2.set_title(f"Perbandingan Volume Saham Aktual vs Prediksi {timeframe_display}")
        # ax2.set_xlabel("Tanggal")
        # ax2.set_ylabel("Volume")

        # Plot untuk High
        high_index = columns.index('High') if 'High' in columns else 2
        ax3.plot(test_dates, y_test_denorm[:, high_index], label="Actual High", color='green')
        ax3.plot(test_dates, predictions_denorm[:, high_index], label="Predicted High", color='orange', linestyle='dashed')
        if isinstance(test_dates.iloc[0], pd.Timestamp):
            ax3.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        ax3.legend()
        ax3.set_title(f"Perbandingan High Saham Aktual vs Prediksi {timeframe_display}")
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
        ax4.set_title(f"Perbandingan Open Saham Aktual vs Prediksi {timeframe_display}")
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
        ax5.set_title(f"Perbandingan Low Saham Aktual vs Prediksi {timeframe_display}")
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
        # volume_errors = calculate_errors(y_test_denorm[:, volume_index], predictions_denorm[:, volume_index])
        high_errors = calculate_errors(y_test_denorm[:, high_index], predictions_denorm[:, high_index])
        open_errors = calculate_errors(y_test_denorm[:, open_index], predictions_denorm[:, open_index])
        low_errors = calculate_errors(y_test_denorm[:, low_index], predictions_denorm[:, low_index])

        # Hitung nilai rata-rata aktual untuk perbandingan relatif
        actual_close_avg = np.mean(np.abs(y_test_denorm[:, close_index]))
        actual_open_avg = np.mean(np.abs(y_test_denorm[:, open_index]))
        actual_high_avg = np.mean(np.abs(y_test_denorm[:, high_index]))
        actual_low_avg = np.mean(np.abs(y_test_denorm[:, low_index]))
        # actual_volume_avg = np.mean(np.abs(y_test_denorm[:, volume_index]))
        
        error_analysis = {
            "Close": close_errors,
            # "Volume": volume_errors,
            "High": high_errors,
            "Open": open_errors,
            "Low": low_errors
        }
        
        # Analisis tren harga saham dengan informasi timeframe
        trend_message, trend = analyze_price_trend(
            y_test_denorm[:, close_index], 
            predictions_denorm[:, close_index],
            prediction_timeframe  # Mengirimkan timeframe yang dipilih user
        )

        # Tambahkan informasi timeframe ke pesan trend
        trend_message = f"Analisis Tren Harga Saham {timeframe_display}: {trend_message}"

        # Di bagian akhir fungsi predict
        if cleanup_after:
            cleanup_result = cleanup_model_files()
            cleanup_message = f"✅ Berhasil menghapus {len(cleanup_result['files_deleted'])} file." if cleanup_result["success"] else "❌ Gagal menghapus beberapa file."
        else:
            cleanup_message = None

        return render_template('prediction.html',
                              stock_name=stock_name, 
                              predictions_html=predictions_html,
                              predictions_csv=predictions_csv, 
                              graph_url=graph_url, 
                              error_analysis=error_analysis,
                              trend_message=trend_message,
                              trend=trend,
                              timeframe=timeframe_display,
                              actual_close_avg=actual_close_avg,
                              actual_open_avg=actual_open_avg,
                              actual_high_avg=actual_high_avg,
                              actual_low_avg=actual_low_avg,
                            #   actual_volume_avg=actual_volume_avg,
                              cleanup_message=cleanup_message)
    return render_template('predict_options.html')

@app.route('/direct_predict', methods=['GET', 'POST'])
def direct_predict():
    if request.method == 'GET':
        return render_template('direct_predict.html')
    
    # Handle POST request
    try:
        # 1. Handle file upload
        if 'model_file' not in request.files:
            return "❌ File model tidak ditemukan!", 400
        
        file = request.files['model_file']
        if file.filename == '' or not file.filename.endswith('.zip'):
            return "❌ File model tidak valid! Harap upload file .zip", 400
        
        # 2. Get form data
        stock_name = request.form['stock_name']
        start_date = request.form['start_date']
        end_date = request.form['end_date']
        prediction_timeframe = request.form.get('prediction_timeframe', 'daily')
        cleanup_after = request.form.get('cleanup', 'false') == 'true'
        
        # 3. Save and extract model
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        
        # Extract zip file
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
                return f"❌ File {req_file} tidak ditemukan dalam model!", 400
        
        # 4. Download stock data
        interval = '1d'
        if prediction_timeframe == 'monthly':
            interval = '1mo'
        elif prediction_timeframe == 'weekly':
            interval = '1wk'
        
        df = yf.download(stock_name, start=start_date, end=end_date, interval=interval)
        if df.empty:
            return "❌ Tidak dapat mengambil data saham. Periksa kode saham dan koneksi internet!", 400
        
        df.reset_index(inplace=True)
        
        # Clean up dataframe
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        df.columns.name = None
        
        # 5. Load model and parameters
        model_path = os.path.join(MODEL_DIR, "lstm_model.h5")
        scaler_path = os.path.join(MODEL_DIR, "scalers.pkl")
        params_path = os.path.join(MODEL_DIR, "model_params.json")
        
        model = load_model(model_path)
        scalers = joblib.load(scaler_path)
        
        with open(params_path, "r") as f:
            model_params = json.load(f)
            time_step = model_params.get('time_step', 5)
        
        # 6. Prepare data for prediction
        processed_df = df.copy()
        if 'Date' in processed_df.columns:
            date_column = processed_df['Date'].copy()
            processed_df = processed_df.drop(columns=['Date'])
        
        # Scale the data
        scaled_data = pd.DataFrame()
        for col in processed_df.columns:
            if col in scalers:
                scaled_data[col] = scalers[col].transform(processed_df[col].values.reshape(-1, 1)).flatten()
            else:
                return f"❌ Kolom {col} tidak ditemukan dalam scaler model!", 400
        
        # 7. Create sequences and predict
        if len(scaled_data) <= time_step:
            return "❌ Data tidak cukup untuk prediksi. Gunakan rentang waktu yang lebih panjang!", 400
        
        # Use the entire dataset for prediction
        X_data, Y_data = create_dataset(scaled_data.values.astype(np.float32), time_step)
        predictions = model.predict(X_data)
        
        # 8. Denormalize predictions and actual values
        columns = list(scalers.keys())
        predictions_denorm = np.zeros_like(predictions)
        y_data_denorm = np.zeros_like(Y_data)
        
        for i, col in enumerate(columns):
            if i < predictions.shape[1]:
                predictions_denorm[:, i] = scalers[col].inverse_transform(predictions[:, i].reshape(-1, 1)).flatten()
                y_data_denorm[:, i] = scalers[col].inverse_transform(Y_data[:, i].reshape(-1, 1)).flatten()
        
        # 9. Prepare results dataframe
        result_df = pd.DataFrame()
        
        for i, col in enumerate(columns):
            if i < predictions.shape[1]:
                result_df[f'{col}'] = y_data_denorm[:, i]
                result_df[f'Pred_{col}'] = predictions_denorm[:, i]
        
        # Add dates if available
        if 'Date' in df.columns:
            test_dates = df['Date'].iloc[time_step:].reset_index(drop=True)
            if len(test_dates) > len(Y_data):
                test_dates = test_dates[:len(Y_data)]
            
            if isinstance(test_dates.iloc[0], pd.Timestamp):
                formatted_dates = test_dates.dt.strftime('%d-%m-%Y')
                result_df.insert(0, "Tanggal", formatted_dates)
            else:
                result_df.insert(0, "Tanggal", test_dates)
        
        result_df.insert(0, "No", range(1, len(result_df) + 1))
        result_df = result_df.round(2)
        
        # 10. Create visualization and analysis (similar to existing predict function)
        predictions_html = result_df.to_html(classes="table table-bordered text-center", border=0, index=False)
        predictions_html = predictions_html.replace('<thead>', '<thead class="table-dark text-center">')
        
        # Save predictions to CSV
        predictions_csv = "static/predictions.csv"
        result_df.to_csv(predictions_csv, index=False)
        
        # Create plots (reuse existing plotting code)
        timeframe_display = get_timeframe_display(prediction_timeframe)
        
        # Get test dates for plotting
        if 'Date' in df.columns:
            test_dates_plot = df['Date'].iloc[time_step:].reset_index(drop=True)
            if len(test_dates_plot) > len(Y_data):
                test_dates_plot = test_dates_plot[:len(Y_data)]
        else:
            test_dates_plot = pd.Series(range(len(Y_data)))
        
        # Create plots (similar to existing code)
        fig, (ax1, ax3, ax4, ax5) = plt.subplots(4, 1, figsize=(14, 25))
        
        # Plot formatting based on timeframe
        if prediction_timeframe == 'daily':
            date_formatter = plt.matplotlib.dates.DateFormatter('%d-%m-%Y')
        elif prediction_timeframe == 'weekly':
            date_formatter = plt.matplotlib.dates.DateFormatter('%d-%m-%Y')
        elif prediction_timeframe == 'monthly':
            date_formatter = plt.matplotlib.dates.DateFormatter('%b %Y')
        else:
            date_formatter = plt.matplotlib.dates.DateFormatter('%d-%m-%Y')
        
        # Plot Close
        close_index = columns.index('Close') if 'Close' in columns else 3
        ax1.plot(test_dates_plot, y_data_denorm[:, close_index], label="Actual Close", color='blue')
        ax1.plot(test_dates_plot, predictions_denorm[:, close_index], label="Predicted Close", color='red', linestyle='dashed')
        if isinstance(test_dates_plot.iloc[0], pd.Timestamp):
            ax1.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.legend()
        ax1.set_title(f"Perbandingan Harga Saham (Close) Aktual vs Prediksi {timeframe_display}")
        ax1.set_xlabel("Tanggal")
        ax1.set_ylabel("Harga Saham")

        # Plot High
        high_index = columns.index('High') if 'High' in columns else 2
        ax3.plot(test_dates_plot, y_data_denorm[:, high_index], label="Actual High", color='green')
        ax3.plot(test_dates_plot, predictions_denorm[:, high_index], label="Predicted High", color='orange', linestyle='dashed')
        if isinstance(test_dates_plot.iloc[0], pd.Timestamp):
            ax3.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        ax3.legend()
        ax3.set_title(f"Perbandingan High Saham Aktual vs Prediksi {timeframe_display}")
        ax3.set_xlabel("Tanggal")
        ax3.set_ylabel("High")

        # Plot Open
        open_index = columns.index('Open') if 'Open' in columns else 1
        ax4.plot(test_dates_plot, y_data_denorm[:, open_index], label="Actual Open", color='green')
        ax4.plot(test_dates_plot, predictions_denorm[:, open_index], label="Predicted Open", color='orange', linestyle='dashed')
        if isinstance(test_dates_plot.iloc[0], pd.Timestamp):
            ax4.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        ax4.legend()
        ax4.set_title(f"Perbandingan Open Saham Aktual vs Prediksi {timeframe_display}")
        ax4.set_xlabel("Tanggal")
        ax4.set_ylabel("Open")

        # Plot Low
        low_index = columns.index('Low') if 'Low' in columns else 4
        ax5.plot(test_dates_plot, y_data_denorm[:, low_index], label="Actual Low", color='green')
        ax5.plot(test_dates_plot, predictions_denorm[:, low_index], label="Predicted Low", color='orange', linestyle='dashed')
        if isinstance(test_dates_plot.iloc[0], pd.Timestamp):
            ax5.xaxis.set_major_formatter(date_formatter)
            plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45)
        ax5.legend()
        ax5.set_title(f"Perbandingan Low Saham Aktual vs Prediksi {timeframe_display}")
        ax5.set_xlabel("Tanggal")
        ax5.set_ylabel("Low")

        # Save graph
        img = io.BytesIO()
        plt.tight_layout()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()

        # Calculate errors
        close_errors = calculate_errors(y_data_denorm[:, close_index], predictions_denorm[:, close_index])
        high_errors = calculate_errors(y_data_denorm[:, high_index], predictions_denorm[:, high_index])
        open_errors = calculate_errors(y_data_denorm[:, open_index], predictions_denorm[:, open_index])
        low_errors = calculate_errors(y_data_denorm[:, low_index], predictions_denorm[:, low_index])

        error_analysis = {
            "Close": close_errors,
            "High": high_errors,
            "Open": open_errors,
            "Low": low_errors
        }

        # Trend analysis
        trend_message, trend = analyze_price_trend(
            y_data_denorm[:, close_index], 
            predictions_denorm[:, close_index],
            prediction_timeframe
        )
        trend_message = f"Analisis Tren Harga Saham {timeframe_display}: {trend_message}"

        # Calculate averages
        actual_close_avg = np.mean(np.abs(y_data_denorm[:, close_index]))
        actual_open_avg = np.mean(np.abs(y_data_denorm[:, open_index]))
        actual_high_avg = np.mean(np.abs(y_data_denorm[:, high_index]))
        actual_low_avg = np.mean(np.abs(y_data_denorm[:, low_index]))

        # Cleanup if requested
        cleanup_message = None
        if cleanup_after:
            cleanup_result = cleanup_model_files()
            cleanup_message = f"✅ Berhasil menghapus {len(cleanup_result['files_deleted'])} file." if cleanup_result["success"] else "❌ Gagal menghapus beberapa file."

        return render_template('prediction.html',
                              stock_name=stock_name, 
                              predictions_html=predictions_html,
                              predictions_csv=predictions_csv, 
                              graph_url=graph_url, 
                              error_analysis=error_analysis,
                              trend_message=trend_message,
                              trend=trend,
                              timeframe=timeframe_display,
                              actual_close_avg=actual_close_avg,
                              actual_open_avg=actual_open_avg,
                              actual_high_avg=actual_high_avg,
                              actual_low_avg=actual_low_avg,
                              cleanup_message=cleanup_message)
                              
    except Exception as e:
        return f"❌ Error dalam prediksi: {str(e)}", 500

@app.route('/cleanup', methods=['GET'])
def cleanup():
    try:
        # Tambahkan clear session TensorFlow untuk membersihkan memori GPU
        from tensorflow.keras import backend as K
        K.clear_session()
    
        result = cleanup_model_files()
        
        if result["success"]:
            message = f"✅ Berhasil membersihkan {len(result['files_deleted'])} file/direktori."
            if result["errors"]:
                message += f" Namun, ada {len(result['errors'])} error."
        else:
            message = "❌ Gagal menghapus file. Lihat detail di konsol server."
        
        # Print detail hasil untuk debugging
        print("\n===== HASIL CLEANUP =====")
        print(f"Files deleted: {len(result['files_deleted'])}")
        print(f"Errors: {len(result['errors'])}")
        if result["errors"]:
            print("\nDetail errors:")
            for err in result["errors"]:
                print(f"- {err}")
        print("========================\n")
        
        # Import garbage collector dan jalankan
        import gc
        gc.collect()
        
        return message
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in cleanup route: {str(e)}")
        print(error_trace)
        return f"❌ Terjadi kesalahan saat cleanup: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
