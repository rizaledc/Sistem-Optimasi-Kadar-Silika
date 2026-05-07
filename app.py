import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pso_logic import PSOOptimizer

# Konfigurasi Tampilan Web
st.set_page_config(page_title="Mining Optimization Dashboard", layout="wide")

# Custom CSS untuk gaya Enterprise
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    
    h1, h2, h3 {
        color: #1e293b; /* Slate 800 */
    }
    
    .stButton>button {
        background-color: #1e3a8a; /* Navy Blue */
        color: white;
        border-radius: 6px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button:hover {
        background-color: #1e40af;
        color: white;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stMetricValue"] {
        color: #10b981; /* Emerald Green */
        font-weight: 700;
    }
    
    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border: 0;
        border-top: 1px solid #e2e8f0; /* Slate 200 */
    }
</style>
""", unsafe_allow_html=True)

# Header Utama
st.markdown("<h1>Sistem Optimasi Kadar Silika</h1>", unsafe_allow_html=True)
st.markdown("<h3>Flotation Plant - Enterprise Dashboard</h3>", unsafe_allow_html=True)
st.markdown("Aplikasi berbasis **Surrogate Model (Random Forest)** dan **Particle Swarm Optimization (PSO)** untuk penentuan parameter mesin pabrik dengan tingkat efisiensi maksimal.")
st.markdown("<hr>", unsafe_allow_html=True)

# Definisi Batas Operasional Mesin (Berdasarkan Dataset)
feature_names = [
    "% Iron Feed", "% Silica Feed", "Starch Flow", "Amina Flow", "Ore Pulp Flow", 
    "Ore Pulp pH", "Ore Pulp Density", "Flotation Column 01 Air Flow", 
    "Flotation Column 02 Air Flow", "Flotation Column 03 Air Flow", 
    "Flotation Column 04 Air Flow", "Flotation Column 05 Air Flow", 
    "Flotation Column 06 Air Flow", "Flotation Column 07 Air Flow", 
    "Flotation Column 01 Level", "Flotation Column 02 Level", 
    "Flotation Column 03 Level", "Flotation Column 04 Level", 
    "Flotation Column 05 Level", "Flotation Column 06 Level", 
    "Flotation Column 07 Level"
]

# Batas bawah dan atas
bounds_min = np.array([48.0, 0.6, 0.0, 240.0, 370.0, 8.5, 1.0, 175.0, 175.0, 175.0, 210.0, 210.0, 180.0, 180.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0, 140.0])
bounds_max = np.array([67.0, 33.0, 6000.0, 750.0, 420.0, 11.0, 1.8, 310.0, 310.0, 310.0, 360.0, 360.0, 330.0, 330.0, 870.0, 870.0, 870.0, 870.0, 870.0, 870.0, 870.0])

try:
    # Memuat logika optimasi
    optimizer = PSOOptimizer('models/rf_surrogate_model.pkl', bounds_min, bounds_max, feature_names)
    
    # Sidebar
    with st.sidebar:
        st.image("assets/logo.png", width=160)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**PENGATURAN ALGORITMA**")
        
        with st.container():
            particles = st.slider("Jumlah Partikel (Swarm Size)", 10, 100, 30)
            iterations = st.slider("Maksimum Iterasi", 10, 100, 50)
            st.markdown("<span style='color: #64748b; font-size: 0.85em;'>Partikel yang lebih banyak meningkatkan presisi eksplorasi ruang komputasi namun menambah beban pemrosesan.</span>", unsafe_allow_html=True)
            
        st.markdown("<hr>", unsafe_allow_html=True)
        run_btn = st.button("JALANKAN OPTIMASI", width='stretch')

    # Layout Tabs
    tab1, tab2, tab3 = st.tabs(["Dashboard Optimasi", "Analisis Parameter", "Panduan Operasional"])
    
    with tab1:
        if run_btn:
            with st.spinner("Memproses algoritma Particle Swarm Optimization..."):
                best_pos, best_val, history = optimizer.optimize(n_particles=particles, n_iterations=iterations)
                
            # Top Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Status Optimasi", value="Selesai", delta="Success", delta_color="normal")
            with col2:
                st.metric(label="Estimasi Kadar Silika Optimal", value=f"{best_val:.4f}%", delta="Target Tercapai", delta_color="inverse")
            with col3:
                st.metric(label="Total Iterasi Konvergensi", value=f"{len(history)}", delta="Efisiensi Tinggi", delta_color="normal")
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Chart and Table Layout
            col_chart, col_table = st.columns([1.2, 1])
            
            with col_chart:
                st.markdown("### Kurva Konvergensi PSO")
                st.markdown("<span style='color: #64748b; font-size: 0.9em;'>Penurunan estimasi kadar silika sepanjang iterasi algoritma.</span>", unsafe_allow_html=True)
                
                fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
                fig.patch.set_alpha(0)
                ax.set_facecolor('none')
                
                # Plot dengan gaya lebih bersih
                ax.plot(history, color='#1e3a8a', linewidth=2.5, marker='o', markersize=4, markerfacecolor='#10b981', markeredgecolor='#10b981')
                
                # Menghilangkan frame atas dan kanan
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_color('#cbd5e1')
                ax.spines['bottom'].set_color('#cbd5e1')
                
                ax.tick_params(colors='#475569')
                ax.set_xlabel("Iterasi ke-", color='#64748b', fontweight='bold')
                ax.set_ylabel("Kadar Silika (%)", color='#64748b', fontweight='bold')
                ax.grid(True, linestyle='--', alpha=0.4, color='#e2e8f0')
                
                st.pyplot(fig)
                
            with col_table:
                st.markdown("### Rekomendasi Setelan Mesin")
                st.markdown("<span style='color: #64748b; font-size: 0.9em;'>Parameter optimal yang ditemukan oleh sistem.</span>", unsafe_allow_html=True)
                
                res_df = pd.DataFrame({
                    "Parameter Operasional": feature_names, 
                    "Nilai Rekomendasi": np.round(best_pos, 4)
                })
                # Tabel Streamlit bawaan yang otomatis mengambil efek bayangan dari CSS
                st.dataframe(res_df, width='stretch', hide_index=True)

        else:
            st.info("Sistem standby. Silakan atur parameter di sidebar dan klik 'JALANKAN OPTIMASI' untuk memulai proses komputasi.")

    with tab2:
        st.markdown("### Analisis Posisi Parameter Terhadap Kapasitas Mesin")
        if run_btn:
            st.write("Grafik di bawah ini memvisualisasikan rekomendasi setelan mesin terhadap rentang batas aman operasionalnya. **0%** berarti parameter disetel sangat rendah (batas bawah minimum), sedangkan **100%** berarti parameter ditekan hingga batas atas maksimum.")
            
            # Menghitung persentase posisi dari batas minimum ke batas maksimum
            range_diff = bounds_max - bounds_min
            range_diff[range_diff == 0] = 1e-9 # Mencegah pembagian nol
            pos_percentage = ((best_pos - bounds_min) / range_diff) * 100
            
            # Persiapan DataFrame untuk plotting
            analysis_df = pd.DataFrame({
                "Parameter": feature_names,
                "Persentase": pos_percentage
            }).sort_values(by="Persentase", ascending=True)
            
            fig_bar, ax_bar = plt.subplots(figsize=(10, 8), dpi=100)
            fig_bar.patch.set_alpha(0)
            ax_bar.set_facecolor('none')
            
            # Tentukan warna: ekstrem tinggi (hijau), ekstrem rendah (abu-abu), menengah (biru navy)
            colors = ['#10b981' if p >= 85 else '#64748b' if p <= 15 else '#1e3a8a' for p in analysis_df["Persentase"]]
            
            bars = ax_bar.barh(analysis_df["Parameter"], analysis_df["Persentase"], color=colors, height=0.6)
            
            # Kustomisasi estetika grafik
            ax_bar.set_xlim(0, 100)
            ax_bar.set_xlabel("Utilisasi / Posisi Parameter (%)", color='#64748b', fontweight='bold')
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            ax_bar.spines['left'].set_color('#cbd5e1')
            ax_bar.spines['bottom'].set_color('#cbd5e1')
            ax_bar.tick_params(colors='#475569')
            ax_bar.grid(True, axis='x', linestyle='--', alpha=0.4, color='#e2e8f0')
            
            st.pyplot(fig_bar)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("💡 **Insight:** Parameter dengan bar berwarna **Hijau** mengindikasikan komponen mesin yang perlu bekerja di kapasitas puncaknya untuk mencapai efisiensi silika optimal. Sedangkan bar **Abu-abu** adalah komponen yang bisa disetel santai/minimum.")
        else:
            st.info("Sistem standby. Silakan atur parameter di sidebar dan klik 'JALANKAN OPTIMASI' terlebih dahulu untuk mengekstrak dan memvisualisasikan data analisis mesin.")
        
    with tab3:
        st.markdown("### Panduan Operasional")
        st.markdown("""
        **1. Persiapan Simulasi**
        - Atur jumlah partikel dan iterasi pada panel sebelah kiri.
        - Partikel merepresentasikan jumlah titik pencarian dalam ruang parameter mesin.
        - Iterasi adalah batas waktu komputasi algoritma pencarian.
        
        **2. Analisis Hasil**
        - Perhatikan metrik **Estimasi Kadar Silika Optimal**. Nilai yang lebih rendah menunjukkan efisiensi pemisahan yang lebih baik.
        - Terapkan **Rekomendasi Setelan Mesin** pada sistem kendali (DCS) secara bertahap.
        
        **3. Catatan Keselamatan**
        - Nilai rekomendasi adalah hasil perhitungan secara matematis murni.
        - Pastikan batas ambang operasional tetap dipantau oleh insinyur kontrol pabrik sebelum menerapkan pada mesin fisik.
        """)

except Exception as e:
    st.error(f"Kegagalan Sistem: {e}")
    st.markdown("Pastikan file model `models/rf_surrogate_model.pkl` tersedia dan valid.")