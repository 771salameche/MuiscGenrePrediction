### MuiscGenrePrediction 
![_ce759ccf-a0be-4900-95d7-c00021239c6f](https://github.com/771salameche/MuiscGenrePrediction/assets/123024504/28841f78-e168-494a-9f4e-ab2b1aa3d987)


### Objectifs
L'objectif principal de ce projet est de créer un système capable de prédire avec précision le genre musical d'un fichier audio. Pour atteindre cet objectif, le projet explore différentes approches de machine learning et d'apprentissage profond, en mettant l'accent sur la spécialisation dans les genres musicaux marocains. Outre la prédiction, le projet vise également à proposer une interface utilisateur conviviale pour faciliter l'interaction avec le système.

### Sources de Données
Les données utilisées pour entraîner et valider les modèles proviennent de sources audio marocaines. La collecte des données s'est faite via des outils open-source comme SpotDL, qui permet de télécharger des fichiers audio de haute qualité à partir de plateformes populaires. Les fichiers audio ont ensuite été segmentés pour garantir une longueur cohérente entre les exemples, ce qui facilite le prétraitement.

### Prétraitement des Données
Avant l'entraînement des modèles, les fichiers audio ont été convertis en format WAV pour une compatibilité optimale avec les bibliothèques de traitement de l'audio comme Librosa. Les caractéristiques audio ont été extraites, notamment les coefficients MFCC, le tempo, le taux de passage par zéro, les caractéristiques RMS, le rolloff spectral, et le centroïde spectral. Ces caractéristiques ont été utilisées comme entrées pour les modèles de machine learning.

### Modèles Utilisés
Le projet utilise deux principaux modèles de machine learning :
1. **CNN** : Un réseau de neurones convolutif composé de couches de convolution, de pooling, de normalisation par lots, et de couches denses. Le CNN est conçu pour extraire des caractéristiques importantes des données audio et effectuer des prédictions sur les genres musicaux.
2. **RNN-LSTM** : Un réseau de neurones récurrents avec des couches LSTM, conçu pour traiter des séquences de données comme des fichiers audio. Le RNN-LSTM peut capturer des dépendances temporelles dans les données, ce qui est crucial pour l'analyse de la musique.

### Validation des Modèles
Les modèles ont été validés en utilisant des métriques de performance telles que l'exactitude, la précision, le rappel, et le F1-score. Les résultats obtenus lors des tests démontrent que le modèle RNN-LSTM a atteint des niveaux de performance élevés, avec une précision supérieure à 99%. Le modèle CNN a également montré des performances solides, avec des valeurs élevées pour le rappel et le F1-score.

### Déploiement de l'Application
L'application a été déployée en utilisant Streamlit, un outil qui permet de créer rapidement des applications web interactives. Les utilisateurs peuvent télécharger des fichiers audio, écouter les extraits, et obtenir des prédictions sur le genre musical. Des éléments de visualisation tels que des matrices de confusion ou des courbes ROC peuvent être ajoutés pour fournir des insights supplémentaires sur les performances des modèles.

### Leçons Tirées et Perspectives Futures
Le projet a démontré l'efficacité des modèles de machine learning pour la classification des genres musicaux. Les principales leçons tirées incluent l'importance d'un prétraitement minutieux des données et le besoin de modèles adaptés aux spécificités des fichiers audio. Pour les développements futurs, le projet envisage d'intégrer davantage de genres musicaux et de raffiner les modèles pour améliorer la précision des prédictions.
