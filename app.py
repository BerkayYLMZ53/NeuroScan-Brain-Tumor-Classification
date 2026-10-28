from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
from PIL import Image

# Eğitilmiş modelleri yüklemek için
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def modelleri_yukle():
    cnn = tf.keras.models.load_model(
        BASE_DIR / "custom_cnn.keras"
    )

    mobilenet = tf.keras.models.load_model(
        BASE_DIR / "mobilenet_model.keras"
    )

    return cnn, mobilenet

# MR görüntüsünü model için hazırlıyoruz
def resmi_hazirla(dosya):

    # Görüntüyü açıyoruz
    resim = Image.open(dosya).convert("RGB")

    # Görüntüyü sayısal diziye dönüştürüyoruz
    resim = np.array(resim)

    # TensorFlow ile boyutlandırıyoruz
    resim = tf.image.resize(resim, (224, 224))

    # Tek görüntülük grup oluşturuyoruz
    resim = tf.expand_dims(resim, axis=0)

    return resim


# Sayfa ayarları
st.set_page_config(
    page_title="NeuroScan",
    page_icon="🧠",
    layout="wide"
)

# Koyu tema ve tasarım
st.markdown("""
<style>
.stApp {
    background-color: #0B1527;
    color: #F1F5FC;
}

h1, h2, h3 {
    color: #56C6F2 !important;
}

/* Normal metinler */
.stApp p {
    color: #E2E8F0;
}

/* İstatistik kartları */
div[data-testid="stMetric"] {
    background-color: #172840;
    border: 1px solid #28435D;
    padding: 20px;
    border-radius: 15px;
}

/* Kart başlıkları */
div[data-testid="stMetricLabel"] p {
    color: #CBD5E1 !important;
}

/* Kart değerleri */
div[data-testid="stMetricValue"] {
    color: #71D8E7;
}

/* Bilgilendirme kutusu */
div[data-testid="stAlert"] {
    background-color: #173343;
    border-radius: 10px;
}

div[data-testid="stAlert"] p {
    color: #D9F4F0 !important;
}

/* Sol menü */
section[data-testid="stSidebar"] {
    background-color: #111F34;
}
</style>
""", unsafe_allow_html=True)

# Başlık alanı
st.title("🧠 NeuroScan")
st.subheader("Beyin MR Analiz Sistemi")

st.write(
    "Derin öğrenme tabanlı beyin MR görüntüsü "
    "sınıflandırma ve klinik karar destek prototipi."
)

st.divider()


# Sol taraftaki gezinme menüsü
with st.sidebar:

    st.title("🧠 NeuroScan")

    st.write("YBS469 Dönem Projesi")

    st.divider()

    sayfa = st.radio(
        "Menü",
        [
            "MR Analizi",
            "Model Karşılaştırması",
            "Proje Hakkında"
        ]
    )


# Eğitilmiş yapay zeka modellerini yüklüyoruz
cnn = None
mobilenet = None

try:
    cnn, mobilenet = modelleri_yukle()
    st.sidebar.success("Yapay zeka modelleri hazır!")

except Exception as hata:
    st.sidebar.error("Modeller yüklenemedi!")
    st.sidebar.write(str(hata))


# Seçilen menüye göre sayfa içeriği



if sayfa == "MR Analizi":

    st.header("MR Görüntüsü Analizi")

    st.write(
        "Analiz etmek istediğiniz beyin MR "
        "görüntüsünü sisteme yükleyiniz."
    )

    
    # Kullanıcının seçtiği modeli saklıyoruz
    secilen_model = st.selectbox(
        "Analiz Modelini Seçiniz",
        ["Custom CNN", "MobileNetV2"],
        key="model_secimi"
    )


    st.write("Seçilen model:", secilen_model)

    
    # MR görüntüsü yükleme alanı
    yuklenen_dosya = st.file_uploader(
        "MR Görüntüsü Seçiniz",
        type=["jpg", "jpeg", "png"],
        key="mr_dosyasi"
    )


    # Görüntü yüklendiyse ekranda göster
    
    if yuklenen_dosya is not None:

        st.success("MR görüntüsü başarıyla yüklendi!")

        # Görüntü ve sonuç için iki sütun
        sol, sag = st.columns(2)

        with sol:
            st.subheader("MR Görüntüsü")
            st.image(yuklenen_dosya, width=350)

        with sag:
            st.subheader("Analiz Sonucu")

            if st.button(
                "🔍 MR Görüntüsünü Analiz Et",
                disabled=(cnn is None or mobilenet is None)
            ):

                hazir_resim = resmi_hazirla(yuklenen_dosya)

                if secilen_model == "Custom CNN":
                    model = cnn
                    test_basarisi = 69.19
                else:
                    model = mobilenet
                    test_basarisi = 91.69

                tahmin = model.predict(hazir_resim, verbose=0)

                tahmin_index = np.argmax(tahmin[0])

                siniflar = [
                    "glioma", "meningioma",
                    "notumor", "pituitary"
                ]

                sonuc = siniflar[tahmin_index]
                olasilik = float(tahmin[0][tahmin_index]) * 100

                st.success("Analiz tamamlandı!")
                st.metric("Tahmin Edilen Sınıf", sonuc.title())
                st.metric("Model Olasılığı", f"%{olasilik:.2f}")

                st.write("Kullanılan model:", secilen_model)
                st.write(f"Genel test doğruluğu: %{test_basarisi:.2f}")

                
                st.info(
                    "Bu sonuç, yapay zeka modelinin "
                    "görüntü üzerinde yaptığı sınıflandırmadır. "
                    "Kesin tanı anlamına gelmez."
                )

                # Modelin en yüksek iki olasılığını karşılaştırıyoruz
                sirali_olasiliklar = np.sort(tahmin[0])

                birinci = float(sirali_olasiliklar[-1]) * 100
                ikinci = float(sirali_olasiliklar[-2]) * 100

                fark = birinci - ikinci

                st.write(
                    f"En yüksek iki sınıf arasındaki "
                    f"olasılık farkı: %{fark:.2f}"
                )


                # Dört sınıfın olasılıklarını göster
                st.subheader("Sınıf Olasılıkları")

                for i in range(4):
                    yuzde = float(tahmin[0][i]) * 100
                    st.write(f"{siniflar[i].title()} — %{yuzde:.2f}")
                    st.progress(float(tahmin[0][i]))

    else:
        st.info("Analize başlamak için MR görüntüsü yükleyiniz.")


elif sayfa == "Model Karşılaştırması":

    st.header("Model Karşılaştırması")

    st.write(
        "Custom CNN ve MobileNetV2 modellerinin "
        "test sonuçlarını karşılaştırabilirsiniz."
    )

    # CSV dosyasını okuyoruz
    tablo = pd.read_csv(
        BASE_DIR / "model_karsilastirma_tahmin_süreleri.csv"
    )

    # Accuracy değerlerini gösteriyoruz
    sutun1, sutun2 = st.columns(2)

    with sutun1:
        st.metric(
            "Custom CNN Accuracy",
            f"%{tablo['Accuracy'][0] * 100:.2f}"
        )

    with sutun2:
        st.metric(
            "MobileNetV2 Accuracy",
            f"%{tablo['Accuracy'][1] * 100:.2f}"
        )

    st.subheader("Performans Sonuçları")

    # Sonuç tablosunu gösteriyoruz
    st.dataframe(tablo, use_container_width=True)
    
    # Grafik için metrikleri seçiyoruz
    metrikler = ["Accuracy", "Precision", "Recall", "F1-Score"]

    # Verileri yüzdeye çeviriyoruz
    grafik_verisi = tablo[["Model"] + metrikler].copy()
    grafik_verisi[metrikler] = grafik_verisi[metrikler] * 100

    # Grafik için tabloyu düzenliyoruz
    grafik_verisi = grafik_verisi.melt(
        id_vars="Model",
        var_name="Metrik",
        value_name="Başarı (%)"
    )

    # Karşılaştırma grafiği
    grafik = px.bar(
        grafik_verisi,
        x="Metrik",
        y="Başarı (%)",
        color="Model",
        barmode="group",
        title="Modellerin Performans Karşılaştırması",
        color_discrete_map={
            "Custom CNN": "#56C6F2",
            "MobileNetV2": "#27B59A"
        }
    )

    # Grafiği koyu temaya uygun yapıyoruz
    grafik.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1527",
        plot_bgcolor="#0B1527",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(grafik, use_container_width=True)
    
    # Tahmin sürelerini karşılaştırıyoruz
    st.subheader("Tahmin Süresi Karşılaştırması")

    sure_grafigi = px.bar(
        tablo,
        x="Model",
        y="Tahmin Suresi (ms)",
        color="Model",
        title="Modellerin Ortalama Tahmin Süreleri",
        text="Tahmin Suresi (ms)",
        color_discrete_map={
            "Custom CNN": "#56C6F2",
            "MobileNetV2": "#27B59A"
        }
    )

    # Grafiğin görünümünü ayarlıyoruz
    sure_grafigi.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1527",
        plot_bgcolor="#0B1527",
        showlegend=False,
        yaxis_title="Süre (ms)"
    )

    st.plotly_chart(sure_grafigi, use_container_width=True)
    
    # Modellerin eğitim geçmişlerini okuyoruz
    cnn_egitim = pd.read_csv(BASE_DIR / "cnn_sonuclar.csv")
    mobilenet_egitim = pd.read_csv(BASE_DIR / "mobilenet_sonuclar.csv")

    st.subheader("Modellerin Eğitim Süreci")

    # Hangi modelin grafiğini göstereceğimizi seçiyoruz
    grafik_modeli = st.selectbox(
        "Eğitim grafiği için model seçiniz",
        ["Custom CNN", "MobileNetV2"]
    )

    if grafik_modeli == "Custom CNN":
        egitim_tablosu = cnn_egitim
    else:
        egitim_tablosu = mobilenet_egitim
    
    # Epoch numaralarını oluşturuyoruz
    egitim_tablosu = egitim_tablosu.copy()
    egitim_tablosu["Epoch"] = range(1, len(egitim_tablosu) + 1)

    # Accuracy grafiği
    accuracy_grafigi = px.line(
        egitim_tablosu,
        x="Epoch",
        y=["accuracy", "val_accuracy"],
        title="Eğitim ve Doğrulama Doğruluğu",
        labels={
            "value": "Accuracy",
            "variable": "Veri Türü"
        },
        markers=True
    )

    accuracy_grafigi.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1527",
        plot_bgcolor="#0B1527",
        legend_title_text=""
    )

    
    
    
    # Loss grafiği
    loss_grafigi = px.line(
        egitim_tablosu,
        x="Epoch",
        y=["loss", "val_loss"],
        title="Eğitim ve Doğrulama Kaybı",
        labels={
            "value": "Loss",
            "variable": "Veri Türü"
        },
        markers=True
    )

    loss_grafigi.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1527",
        plot_bgcolor="#0B1527",
        legend_title_text=""
    )

    
    # MobileNetV2 eğitim aşamalarını işaretliyoruz
    if grafik_modeli == "MobileNetV2":

        for grafik in [accuracy_grafigi, loss_grafigi]:

            # İlk 10 epoch sonu
            grafik.add_vline(
                x=10.5,
                line_dash="dash",
                line_color="#FBBF24"
            )

            # Fine-tuning aşamasının sonu
            grafik.add_vline(
                x=15.5,
                line_dash="dash",
                line_color="#F472B6"
            )
    st.plotly_chart(accuracy_grafigi, use_container_width=True)
    st.plotly_chart(loss_grafigi, use_container_width=True)


    
    # Sınıflara göre modellerin başarısı
    st.subheader("Sınıf Bazlı Performans")

    st.write(
        "Bu grafik, modellerin her MR sınıfını "
        "ne oranda doğru tespit ettiğini gösterir."
    )

    # Colab'daki test sonuçlarımız
    sinif_tablosu = pd.DataFrame({
        "Sınıf": [
            "Glioma",
            "Meningioma",
            "No Tumor",
            "Pituitary"
        ],
        "Custom CNN": [55.25, 44.00, 93.00, 84.50],
        "MobileNetV2": [74.50, 94.25, 99.75, 98.25]
    })

    # Tabloyu grafik için düzenliyoruz
    sinif_grafik_verisi = sinif_tablosu.melt(
        id_vars="Sınıf",
        var_name="Model",
        value_name="Recall (%)"
    )

    # Sütun grafiği oluşturuyoruz
    sinif_grafigi = px.bar(
        sinif_grafik_verisi,
        x="Sınıf",
        y="Recall (%)",
        color="Model",
        barmode="group",
        title="Sınıflara Göre Model Başarıları",
        color_discrete_map={
            "Custom CNN": "#56C6F2",
            "MobileNetV2": "#27B59A"
        }
    )

    sinif_grafigi.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0B1527",
        plot_bgcolor="#0B1527",
        yaxis_range=[0, 100]
    )

    st.plotly_chart(sinif_grafigi, use_container_width=True)
    
    # Karmaşıklık matrislerini hazırlıyoruz
    st.subheader("Confusion Matrix")

    sinif_adlari = [
        "Glioma", "Meningioma", "No Tumor", "Pituitary"
    ]

    # Colab'dan elde ettiğimiz gerçek değerler
    cnn_matris = [
        [221, 81, 66, 32],
        [58, 176, 115, 51],
        [9, 5, 372, 14],
        [11, 50, 1, 338]
    ]

    mobilenet_matris = [
        [298, 84, 15, 3],
        [6, 377, 6, 11],
        [0, 1, 399, 0],
        [2, 4, 1, 393]
    ]

    # Kullanıcının incelemek istediği modeli seçiyoruz
    matris_modeli = st.selectbox(
        "Confusion Matrix için model seçiniz",
        ["Custom CNN", "MobileNetV2"]
    )

    if matris_modeli == "Custom CNN":
        matris = cnn_matris
    else:
        matris = mobilenet_matris

    # Matris grafiğini oluşturuyoruz
    cm_grafigi = px.imshow(
        matris,
        x=sinif_adlari,
        y=sinif_adlari,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={
            "x": "Tahmin Edilen Sınıf",
            "y": "Gerçek Sınıf",
            "color": "Görüntü Sayısı"
        }
    )

    cm_grafigi.update_layout(
        title=f"{matris_modeli} - Confusion Matrix",
        template="plotly_dark",
        paper_bgcolor="#0B1527",
        font=dict(color="#E2E8F0")
    )

    st.plotly_chart(cm_grafigi, use_container_width=True)

   


elif sayfa == "Proje Hakkında":

    st.header("Proje Hakkında")

    st.write(
        "Bu proje, YBS469 Yapay Zeka ve Uygulamaları "
        "dersi kapsamında geliştirilmiştir."
    )

    st.subheader("Projenin Amacı")

    st.write(
        "Beyin MR görüntülerini derin öğrenme "
        "algoritmalarıyla dört farklı sınıfa ayırmak "
        "ve elde edilen sonuçları bir karar destek "
        "prototipinde göstermektir."
    )

    st.subheader("Kullanılan Veri Seti")

    st.write("Veri seti: Brain Tumor MRI Dataset")

    st.write("Toplam görüntü sayısı: 7.200")

    st.write(
        "Sınıflar: Glioma, Meningioma, "
        "No Tumor ve Pituitary"
    )

    st.subheader("Veri Dağılımı")

    sutun1, sutun2, sutun3 = st.columns(3)

    with sutun1:
        st.metric("Eğitim", "4.480")

    with sutun2:
        st.metric("Doğrulama", "1.120")

    with sutun3:
        st.metric("Test", "1.600")

    st.subheader("Kullanılan Modeller")

    st.write(
        "**Custom CNN:** Sıfırdan oluşturulan "
        "üç evrişim bloğuna sahip derin öğrenme modeli."
    )

    st.write(
        "**MobileNetV2:** ImageNet ağırlıklarıyla "
        "başlatılan ve sonrasında fine-tuning "
        "uygulanan transfer öğrenme modeli."
    )

    st.subheader("Çalışma Yöntemi")

    st.write(
        "Görüntüler 224x224 piksel boyutuna "
        "getirilerek TensorFlow ile eğitilmiştir. "
        "Modeller Accuracy, Precision, Recall, "
        "F1-Score ve tahmin süresiyle karşılaştırılmıştır."
    )

    st.subheader("Sistemin Sınırlılıkları")

    st.warning(
        "Bu uygulama yalnızca eğitim ve araştırma "
        "amaçlıdır. Tıbbi tanı yerine kullanılamaz. "
        "Model sonuçları uzman değerlendirmesinin "
        "yerini tutmaz."
    )



