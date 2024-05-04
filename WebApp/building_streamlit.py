# -*- coding: utf-8 -*-
"""
Created on Sat May  4 17:09:49 2024

@author: salah
"""

import streamlit as st
import os
import librosa
import numpy as np
import base64
from pydub import AudioSegment
from tensorflow.keras.models import load_model
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Prédiction de Genre Musical",
                   page_icon='🎵',
                   layout='wide',
                   initial_sidebar_state="expanded",)


music_png = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Nuvola_Moroccan_flag.svg/640px-Nuvola_Moroccan_flag.svg.png'
#st.sidebar.image(music_png, width=100, use_column_width=True, output_format='auto')




# Charger les modèles
model1 = load_model(r"C:\Users\salah\model_1.h5")
model2 = load_model(r"C:\Users\salah\modèle_RNN_LSTM_1.h5")

# Initialisation de audio_features
audio_features = None

def custom_css():
    css = """
    <style>
    /* Changer la couleur de la barre latérale */
    .st-emotion-cache-6qob1r.eczjme3 {
    background-color: #1DB954 !important;  # Vert de Spotify
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    
    
    
    
# Fonction pour ajouter du CSS personnalisé avec une image d'arrière-plan
def add_custom_css(background_image_path):
    with open(background_image_path, "rb") as img_file:
        image_data = img_file.read()

    encoded_image = base64.b64encode(image_data).decode()  # Encodage base64

    # Ajout de CSS personnalisé avec une image d'arrière-plan
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

#add_custom_css(r"C:\Users\salah\Downloads\pexels-postiglioni-943535.jpg")






# Liste des genres musicaux 
genres = [
    ('Amazigh (Ahidous)'),
    ('Chaâbi'),
    ('Gnawa'),
    ('Malhun'),
    ('Musique Andalouse'),
    ('Rap et Hip-Hop Marocain'),
    ('Raï'),
    ('Reggada'),
    ('Sufi')
]



# Fonction pour convertir MP3 en WAV
def convert_to_wav(mp3_file_path):
    audio_segment = AudioSegment.from_file(mp3_file_path, format="mp3")
    wav_file_path = os.path.splitext(mp3_file_path)[0] + ".wav"
    audio_segment.export(wav_file_path, format="wav")
    return wav_file_path


def extract_all_features(filename):
    y, sr = librosa.load(filename, duration=30)
    length = len(y)
    
    # Calculer les caractéristiques RMS
    rms_mean = librosa.feature.rms(y=y).mean()
    rms_var = librosa.feature.rms(y=y).var()
    
    # Calculer les caractéristiques du centroïde spectral
    spectral_centroid_mean = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    spectral_centroid_var = librosa.feature.spectral_centroid(y=y, sr=sr).var()
    
    # Calculer les caractéristiques de la largeur de bande spectrale
    spectral_bandwidth_mean_mean = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    spectral_bandwidth_mean_var = librosa.feature.spectral_bandwidth(y=y, sr=sr).var()
    
    # Calculer les caractéristiques du rolloff spectral
    rolloff_mean = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    rolloff_var = librosa.feature.spectral_rolloff(y=y, sr=sr).var()
    
    # Calculer les caractéristiques du taux de passage par zéro
    zero_crossing_rate_mean = librosa.feature.zero_crossing_rate(y=y).mean()
    zero_crossing_rate_var = librosa.feature.zero_crossing_rate(y=y).var()
    
    # Calculer les caractéristiques de l'harmonie
    y_harmonic = librosa.effects.harmonic(y)
    harmony_mean = np.mean(y_harmonic)
    harmony_var = np.var(y_harmonic)
    
    # Calculer les caractéristiques du percussif
    y_percussive = librosa.effects.percussive(y)
    percussive_mean = np.mean(y_percussive)
    percussive_var = np.var(y_percussive)
    
    # Calculer le tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Calculer les coefficients MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1)
    mfcc_vars = np.var(mfccs, axis=1)
    
    # Concaténer toutes les caractéristiques extraites
    feature = np.concatenate(([length], [rms_mean], [rms_var], [spectral_centroid_mean], [spectral_centroid_var],
                              [spectral_bandwidth_mean_mean], [spectral_bandwidth_mean_var], [rolloff_mean],
                              [rolloff_var], [zero_crossing_rate_mean], [zero_crossing_rate_var], [harmony_mean],
                              [harmony_var], [percussive_mean], [percussive_var], [tempo], mfcc_means, mfcc_vars))

    return feature


# Barre latérale
def main():
    global audio_features  # Déclarez audio_features comme variable globale

    # Appliquer le CSS personnalisé
    custom_css()

    with st.sidebar:
    # Définition du menu avec option_menu
        page = option_menu("Menu Principal", ["Page d'accueil", "CNN", "RNN-LSTM", "À propos"],
                           icons=["house", "graph-up-arrow", "graph-up-arrow", "info-circle"], 
                           menu_icon="cast", default_index=0)
    st.sidebar.write("©2024, Developed By KAYOUH Salaheddine")
    if page == "Page d'accueil":
        st.title("Prédiction de Genre Musical avec Deep Learning")

        st.markdown("""
                    ## Bienvenue!
                    Cette application web utilise des modèles d'apprentissage profond pour prédire le genre de vos fichiers audio. Téléchargez simplement votre fichier, et laissez l'intelligence artificielle faire le reste.
    
                    ## Comment ça fonctionne
                    - **Téléchargez** un fichier audio (formats supportés: MP3, WAV).
                    - Attendez que le modèle **analyse le fichier** et **prédise le genre** musical.
                    - **Consultez les résultats** de la prédiction.
    
                    ## Guide d'Utilisation
                    Pour commencer, utilisez le widget de téléchargement de fichiers dans la barre latérale pour charger votre fichier audio. Vous pouvez voir et écouter votre fichier ci-dessous une fois chargé. Ensuite, cliquez sur le bouton de prédiction pour obtenir les résultats.
    
                    ## À Propos du Modèle
                    Cette application utilise des réseaux de neurones CNN (Convolutional Neural Network) et RNN-LSTM (Recurrent Neural Network with Long Short-Term Memory) entraînés exclusivement sur un dataset composé de fichiers audio de genres musicaux 100% marocains. Ces modèles sont conçus pour identifier des nuances subtiles dans les audios afin de prédire avec précision le genre musical.

                    ## Performances du modèle
                    - **Précision de prédiction:** 90%
                    - **Temps de réponse:** Moins de 10 secondes en moyenne par fichier audio.
    
                    ## FAQ
                    **Q: Quelle est la précision du modèle ?**
                    **R:** Le modèle a une précision d'environ 90%, mais cela peut varier selon la complexité du fichier audio.

                    **Q: Puis-je télécharger n'importe quel type de fichier audio ?**
                    **R:** Actuellement, seuls les fichiers MP3 et WAV sont supportés.

                    ## Contact et Feedback
                    Votre feedback est très important pour nous ! N'hésitez pas à [nous contacter](https://www.linkedin.com/in/salaheddine-kayouh-899b34235/) pour tout commentaire ou question.

                    
                                        """)
    elif page == "CNN":
        st.title("Prédiction de Genre Musical avec CNN")
        uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["mp3", "wav"])     
        if uploaded_file:
            # Chemin temporaire
            temp_path = os.path.join("/tmp", uploaded_file.name)

            # Sauvegarder le fichier téléchargé
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Convertir en WAV si nécessaire
            if uploaded_file.name.lower().endswith(".mp3"):
                temp_path = convert_to_wav(temp_path)

            # Afficher le fichier audio pour lecture
            st.audio(uploaded_file, format="audio/mp3")

            # Extraire les caractéristiques audio
            audio_features = extract_all_features(temp_path)

            # Ajuster la forme des caractéristiques audio
            audio_features = np.expand_dims(audio_features, axis=0)
            prediction1 = model1.predict(np.expand_dims(audio_features, axis=0))
            predicted_genre_1 = genres[np.argmax(prediction1)]
            st.success(f"Prédiction avec le premier modèle : {predicted_genre_1}")
    elif page == "RNN-LSTM":
        st.title("Prédiction de Genre Musical avec RNN-LSTM ")
        uploaded_file = st.file_uploader("Choisissez un fichier audio", type=["mp3", "wav"])
        if uploaded_file:
            # Chemin temporaire
            temp_path = os.path.join("/tmp", uploaded_file.name)

            # Sauvegarder le fichier téléchargé
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Convertir en WAV si nécessaire
            if uploaded_file.name.lower().endswith(".mp3"):
                temp_path = convert_to_wav(temp_path)

            # Afficher le fichier audio pour lecture
            st.audio(uploaded_file, format="audio/mp3")

            # Extraire les caractéristiques audio
            audio_features = extract_all_features(temp_path)

            # Ajuster la forme des caractéristiques audio
            audio_features = np.expand_dims(audio_features, axis=0)
            prediction2 = model2.predict(audio_features)
            predicted_genre_2 = genres[np.argmax(prediction2)]
            st.success(f"Prédiction avec le deuxième modèle : {predicted_genre_2}")
    elif page == "À propos":
        st.title('À propos de l\'application')
        st.write("""
    Cette application utilise des techniques avancées d'apprentissage machine pour analyser et prédire 
    le genre de fichiers audio spécifiquement marocains. Elle vise à démontrer comment des modèles de deep learning
    peuvent être appliqués efficacement dans le domaine du traitement audio.
""")
        st.header('Sources de Données')
        st.write("""
        Les données utilisées pour entraîner ces modèles proviennent exclusivement de sources audio marocaines, 
        permettant ainsi au modèle de spécialiser dans les particularités de ce genre musical.
    """)
        st.header('Technologies Utilisées')
        st.write("""
        Les technologies et librairies principales utilisées comprennent:
        - **Python** : Langage de programmation.
        - **Streamlit** : Pour le développement rapide d'applications web.
        - **TensorFlow et Keras** : Pour la construction et l'entraînement des modèles de machine learning.
        - **Librosa** : Pour l'extraction des caractéristiques audio.
    """)
        st.header('Remerciements et Crédits')
        st.write("""
        Un grand merci à tous ceux qui ont contribué au développement de cette application, 
        en particulier à mes collègues et mentors dans le domaine de l'IA.
    """)

        st.header('Futurs Développements')
        st.write("""
        Les développements futurs incluront l'ajout de nouveaux modèles pour améliorer la précision des prédictions
        et l'expansion de la base de données pour inclure d'autres genres audio régionaux.
    """)


if __name__ == "__main__":
    main()
    
    

