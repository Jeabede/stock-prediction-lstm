import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pickle
import io
import zipfile
import tempfile
import os
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Prediksi Saham LQ45 - LSTM",
    page_icon="📈",
    layout="wide"
)

# Dictionary saham LQ45 sektor keuangan
SAHAM_DICT = {
    'BBCA': 'BBCA.JK',
    'BBNI': 'BBNI.JK',
    'BBRI': 'BBRI.JK',
    'BBTN': 'BBTN.JK',
    'BMRI': 'BMRI.JK'
}

# Fungsi untuk mengunduh data saham
@st.cache_data(ttl=3600)
def download_stock_data(ticker, start_date, end_date, interval='1d'):
    """Download data saham dari Yahoo Finance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error mengunduh data: {str(e)}")
        return None

# Fungsi untuk memproses data
def prepare_data(data, train_size=0.8):
    """Memproses dan membagi data menjadi train dan test"""
    # Ambil kolom yang diperlukan
    df = data[['Open', 'High', 'Low', 'Close']].copy()
    
    # Pisahkan data train dan test
    train_len = int(len(df) * train_size)
    train_data = df[:train_len]
    test_data = df[train_len:]
    
    # Normalisasi data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    
    return train_scaled, test_scaled, scaler, train_data, test_data

# Fungsi untuk membuat sequences
def create_sequences(data, seq_length=60):
    """Membuat sequences untuk LSTM"""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 3])  # Prediksi Close price
    return np.array(X), np.array(y)

# Fungsi untuk membangun model LSTM
def build_lstm_model(input_shape, neurons=50):
    """Membangun model LSTM"""
    model = Sequential([
        LSTM(neurons, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(neurons, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Fungsi untuk evaluasi model
def evaluate_model(y_true, y_pred):
    """Menghitung metrik evaluasi"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Hitung persentase error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # F1 Score (untuk klasifikasi naik/turun)
    y_true_class = (np.diff(y_true.flatten()) > 0).astype(int)
    y_pred_class = (np.diff(y_pred.flatten()) > 0).astype(int)
    
    # Tambahkan nilai dummy untuk menghindari error jika hanya satu class
    if len(np.unique(y_true_class)) == 1:
        f1 = 0.0
    else:
        f1 = f1_score(y_true_class, y_pred_class, average='binary', zero_division=0)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'F1_Score': f1
    }

# Fungsi untuk membuat kesimpulan
def generate_conclusion(predictions, original_prices):
    """Membuat kesimpulan prediksi"""
    last_original = original_prices[-1]
    last_prediction = predictions[-1]
    
    change = ((last_prediction - last_original) / last_original) * 100
    
    if change > 2:
        trend = "📈 NAIK"
        color = "green"
        recommendation = "Tren prediksi menunjukkan kenaikan harga yang signifikan."
    elif change < -2:
        trend = "📉 TURUN"
        color = "red"
        recommendation = "Tren prediksi menunjukkan penurunan harga yang signifikan."
    else:
        trend = "➡️ STABIL"
        color = "orange"
        recommendation = "Tren prediksi menunjukkan harga cenderung stabil."
    
    return trend, abs(change), color, recommendation

# Fungsi untuk membuat ZIP file berisi model dan scaler
def create_model_zip(model, scaler, stock_name, train_date):
    """Membuat ZIP file berisi model dan scaler"""
    zip_buffer = io.BytesIO()
    
    # Gunakan temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, 'model.keras')
        scaler_path = os.path.join(temp_dir, 'scaler.pkl')
        metadata_path = os.path.join(temp_dir, 'metadata.pkl')
        
        # Simpan model ke file temporary
        model.save(model_path)
        
        # Simpan scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Simpan metadata
        metadata = {
            'stock_name': stock_name,
            'train_date': train_date,
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        # Buat ZIP file
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.write(model_path, 'model.keras')
            zip_file.write(scaler_path, 'scaler.pkl')
            zip_file.write(metadata_path, 'metadata.pkl')
    
    zip_buffer.seek(0)
    return zip_buffer

# Fungsi untuk ekstrak model dari ZIP
def extract_model_from_zip(zip_file):
    """Ekstrak model, scaler, dan metadata dari ZIP file"""
    try:
        # Gunakan temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Ekstrak semua file ke temporary directory
                zip_ref.extractall(temp_dir)
            
            # Load model
            model_path = os.path.join(temp_dir, 'model.keras')
            model = load_model(model_path)
            
            # Load scaler
            scaler_path = os.path.join(temp_dir, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load metadata
            try:
                metadata_path = os.path.join(temp_dir, 'metadata.pkl')
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
            except:
                metadata = None
            
            return model, scaler, metadata
    except Exception as e:
        raise Exception(f"Error ekstrak ZIP: {str(e)}")

# Halaman utama
def main():
    st.title("📈 Aplikasi Prediksi Harga Saham LQ45 dengan LSTM")
    st.markdown("---")
    
    # Sidebar untuk navigasi
    menu = st.sidebar.selectbox(
        "Menu",
        ["🏗️ Membangun Model LSTM", "🔮 Prediksi Langsung"]
    )
    
    if menu == "🏗️ Membangun Model LSTM":
        build_model_page()
    else:
        prediction_page()

# Halaman membangun model
def build_model_page():
    st.header("🏗️ Membangun Model LSTM")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1️⃣ Pilih Saham")
        selected_stock = st.selectbox(
            "Pilih Saham LQ45 (Sektor Keuangan)",
            list(SAHAM_DICT.keys())
        )
        
        st.subheader("2️⃣ Tentukan Periode")
        start_date = st.date_input(
            "Tanggal Mulai",
            value=datetime.now() - timedelta(days=730)
        )
        end_date = st.date_input(
            "Tanggal Akhir",
            value=datetime.now()
        )
        
        interval = st.selectbox(
            "Interval Data",
            ["1d (Harian)", "1wk (Mingguan)", "1mo (Bulanan)"]
        )
        interval_map = {"1d (Harian)": "1d", "1wk (Mingguan)": "1wk", "1mo (Bulanan)": "1mo"}
        selected_interval = interval_map[interval]
    
    with col2:
        st.subheader("3️⃣ Parameter Model")
        neurons = st.slider("Jumlah Neuron LSTM", 25, 200, 50, 25)
        batch_size = st.slider("Batch Size", 16, 128, 32, 16)
        epochs = st.slider("Epochs", 10, 200, 50, 10)
        
        st.subheader("4️⃣ Aksi")
        save_model = st.checkbox("Simpan Model Setelah Training", value=True)
    
    if st.button("🚀 Mulai Training", type="primary", use_container_width=True):
        with st.spinner("Mengunduh data saham..."):
            ticker = SAHAM_DICT[selected_stock]
            data = download_stock_data(ticker, start_date, end_date, selected_interval)
            
            if data is None or len(data) < 100:
                st.error("Data tidak cukup atau gagal diunduh. Minimal 100 data diperlukan.")
                return
            
            st.success(f"✅ Data berhasil diunduh: {len(data)} baris data")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Prepare data
        status_text.text("Memproses data...")
        progress_bar.progress(20)
        train_scaled, test_scaled, scaler, train_data, test_data = prepare_data(data)
        
        # Create sequences
        status_text.text("Membuat sequences...")
        progress_bar.progress(30)
        seq_length = 60
        X_train, y_train = create_sequences(train_scaled, seq_length)
        X_test, y_test = create_sequences(test_scaled, seq_length)
        
        # Build model
        status_text.text("Membangun model LSTM...")
        progress_bar.progress(40)
        model = build_lstm_model((X_train.shape[1], X_train.shape[2]), neurons)
        
        # Train model
        status_text.text(f"Training model... (Epochs: {epochs})")
        progress_bar.progress(50)
        
        # Callback untuk update progress
        class ProgressCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = 50 + int((epoch + 1) / epochs * 30)
                progress_bar.progress(progress)
                status_text.text(f"Training... Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.6f}")
        
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.1,
            verbose=0,
            callbacks=[ProgressCallback()]
        )
        
        # Predict
        status_text.text("Melakukan prediksi...")
        progress_bar.progress(85)
        
        train_predict = model.predict(X_train, verbose=0)
        test_predict = model.predict(X_test, verbose=0)
        
        # Inverse transform untuk mendapatkan harga asli
        train_predict_prices = scaler.inverse_transform(
            np.concatenate([np.zeros((len(train_predict), 3)), train_predict], axis=1)
        )[:, 3]
        
        test_predict_prices = scaler.inverse_transform(
            np.concatenate([np.zeros((len(test_predict), 3)), test_predict], axis=1)
        )[:, 3]
        
        y_test_prices = scaler.inverse_transform(
            np.concatenate([np.zeros((len(y_test), 3)), y_test.reshape(-1, 1)], axis=1)
        )[:, 3]
        
        progress_bar.progress(100)
        status_text.text("✅ Training selesai!")
        
        st.markdown("---")
        st.header("📊 Hasil Prediksi")
        
        # Evaluasi
        metrics = evaluate_model(y_test_prices, test_predict_prices)
        
        # Tampilkan metrik
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
            st.caption(f"({metrics['MAPE']:.2f}%)")
        
        with col2:
            st.metric("MSE", f"{metrics['MSE']:.2f}")
        
        with col3:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        
        with col4:
            st.metric("F1 Score", f"{metrics['F1_Score']:.4f}")
            st.caption(f"({metrics['F1_Score']*100:.2f}%)")
        
        with col5:
            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        
        # Grafik
        st.subheader("📈 Visualisasi Prediksi vs Aktual")
        
        # Siapkan data untuk plotting
        test_dates = data.index[len(train_data) + seq_length:]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_test_prices,
            mode='lines',
            name='Harga Aktual',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_predict_prices,
            mode='lines',
            name='Harga Prediksi',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Prediksi Harga Saham {selected_stock}',
            xaxis_title='Tanggal',
            yaxis_title='Harga (IDR)',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Kesimpulan
        st.subheader("💡 Kesimpulan Prediksi")
        trend, change_pct, color, recommendation = generate_conclusion(test_predict_prices, y_test_prices)
        
        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {color}20; border-left: 5px solid {color}'>
            <h3 style='color: {color}; margin: 0;'>{trend}</h3>
            <p style='font-size: 18px; margin: 10px 0;'>Perubahan Prediksi: <b>{change_pct:.2f}%</b></p>
            <p style='margin: 0;'>{recommendation}</p>
            <p style='margin-top: 10px; font-size: 14px; color: gray;'>
                <i>⚠️ Catatan: Prediksi ini adalah hasil model machine learning dan tidak menjamin hasil investasi. 
                Selalu lakukan riset mendalam sebelum berinvestasi.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Simpan model
        if save_model:
            st.markdown("---")
            st.subheader("💾 Simpan Model")
            
            # Buat nama file otomatis
            train_date = datetime.now().strftime('%Y%m%d')
            zip_filename = f"{selected_stock}_{train_date}.zip"
            
            # Buat ZIP file
            with st.spinner("Membuat file ZIP..."):
                zip_buffer = create_model_zip(model, scaler, selected_stock, train_date)
            
            st.download_button(
                label="📥 Download Model (ZIP)",
                data=zip_buffer,
                file_name=zip_filename,
                mime="application/zip",
                use_container_width=True
            )
            
            st.success(f"✅ Model siap diunduh: `{zip_filename}`")
            st.info("💡 File ZIP berisi: model.h5, scaler.pkl, dan metadata.pkl")

# Halaman prediksi langsung
def prediction_page():
    st.header("🔮 Prediksi Langsung dengan Model Terlatih")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1️⃣ Upload Model (ZIP)")
        uploaded_zip = st.file_uploader("Upload file ZIP model", type=['zip'])
        
        if uploaded_zip is not None:
            try:
                model, scaler, metadata = extract_model_from_zip(uploaded_zip)
                st.success("✅ Model berhasil dimuat!")
                
                if metadata:
                    st.info(f"📦 Model: **{metadata['stock_name']}** | Training: **{metadata['train_date']}**")
            except Exception as e:
                st.error(f"❌ Error memuat model: {str(e)}")
                model, scaler, metadata = None, None, None
        else:
            model, scaler, metadata = None, None, None
        
        st.subheader("2️⃣ Pilih Saham untuk Prediksi")
        selected_stock = st.selectbox(
            "Pilih Saham",
            list(SAHAM_DICT.keys())
        )
    
    with col2:
        st.subheader("3️⃣ Tentukan Periode")
        start_date = st.date_input(
            "Tanggal Mulai",
            value=datetime.now() - timedelta(days=365)
        )
        end_date = st.date_input(
            "Tanggal Akhir",
            value=datetime.now()
        )
        
        interval = st.selectbox(
            "Interval Data",
            ["1d (Harian)", "1wk (Mingguan)", "1mo (Bulanan)"]
        )
        interval_map = {"1d (Harian)": "1d", "1wk (Mingguan)": "1wk", "1mo (Bulanan)": "1mo"}
        selected_interval = interval_map[interval]
    
    if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):
        if model is None or scaler is None:
            st.error("❌ Harap upload model ZIP terlebih dahulu!")
            return
        
        with st.spinner("Mengunduh data saham..."):
            ticker = SAHAM_DICT[selected_stock]
            data = download_stock_data(ticker, start_date, end_date, selected_interval)
            
            if data is None or len(data) < 100:
                st.error("Data tidak cukup atau gagal diunduh. Minimal 100 data diperlukan.")
                return
            
            st.success(f"✅ Data berhasil diunduh: {len(data)} baris data")
        
        # Prepare data
        train_scaled, test_scaled, _, train_data, test_data = prepare_data(data)
        
        # Create sequences
        seq_length = 60
        X_train, y_train = create_sequences(train_scaled, seq_length)
        X_test, y_test = create_sequences(test_scaled, seq_length)
        
        # Predict
        with st.spinner("Melakukan prediksi..."):
            test_predict = model.predict(X_test, verbose=0)
            
            # Inverse transform
            test_predict_prices = scaler.inverse_transform(
                np.concatenate([np.zeros((len(test_predict), 3)), test_predict], axis=1)
            )[:, 3]
            
            y_test_prices = scaler.inverse_transform(
                np.concatenate([np.zeros((len(y_test), 3)), y_test.reshape(-1, 1)], axis=1)
            )[:, 3]
        
        st.markdown("---")
        st.header("📊 Hasil Prediksi")
        
        # Evaluasi
        metrics = evaluate_model(y_test_prices, test_predict_prices)
        
        # Tampilkan metrik
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
            st.caption(f"({metrics['MAPE']:.2f}%)")
        
        with col2:
            st.metric("MSE", f"{metrics['MSE']:.2f}")
        
        with col3:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        
        with col4:
            st.metric("F1 Score", f"{metrics['F1_Score']:.4f}")
            st.caption(f"({metrics['F1_Score']*100:.2f}%)")
        
        with col5:
            st.metric("MAPE", f"{metrics['MAPE']:.2f}%")
        
        # Grafik
        st.subheader("📈 Visualisasi Prediksi vs Aktual")
        
        test_dates = data.index[len(train_data) + seq_length:]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=y_test_prices,
            mode='lines',
            name='Harga Aktual',
            line=dict(color='blue', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates,
            y=test_predict_prices,
            mode='lines',
            name='Harga Prediksi',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title=f'Prediksi Harga Saham {selected_stock}',
            xaxis_title='Tanggal',
            yaxis_title='Harga (IDR)',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Kesimpulan
        st.subheader("💡 Kesimpulan Prediksi")
        trend, change_pct, color, recommendation = generate_conclusion(test_predict_prices, y_test_prices)
        
        st.markdown(f"""
        <div style='padding: 20px; border-radius: 10px; background-color: {color}20; border-left: 5px solid {color}'>
            <h3 style='color: {color}; margin: 0;'>{trend}</h3>
            <p style='font-size: 18px; margin: 10px 0;'>Perubahan Prediksi: <b>{change_pct:.2f}%</b></p>
            <p style='margin: 0;'>{recommendation}</p>
            <p style='margin-top: 10px; font-size: 14px; color: gray;'>
                <i>⚠️ Catatan: Prediksi ini adalah hasil model machine learning dan tidak menjamin hasil investasi. 
                Selalu lakukan riset mendalam sebelum berinvestasi.</i>
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()