# Rapport de Projet : Détection de Maladies des Cultures

## 1. Contexte et Objectif
La détection précoce des maladies des plantes est un enjeu majeur pour l'agriculture moderne, permettant d'optimiser les rendements et de limiter l'usage de pesticides. L'objectif de ce projet est de développer un modèle d'Intelligence Artificielle basé sur le Deep Learning capable d'identifier automatiquement diverses maladies à partir de photos de feuilles de plantes.

## 2. Données Utilisées
Ce projet s'appuie sur le dataset public **PlantVillage**, un standard dans la communauté pour l'analyse visuelle des feuilles.
- **Nombre de classes** : 38 classes (comprenant différentes paires de type de plante / maladie, ainsi que des feuilles saines).
- **Prétraitement et Data Augmentation** : Afin d'améliorer la robustesse du modèle (réduction de l'overfitting), le jeu d'entraînement bénéficie de techniques de data augmentation (Random Resized Crop, Random Horizontal Flip, Color Jitter). Les images sont redimensionnées et normalisées selon les standards ImageNet.
- **Répartition** : Le jeu de données est divisé en trois sous-ensembles : 80% pour l'entraînement, 10% pour la validation et 10% pour le test.

## 3. Architecture du Modèle
L'approche choisie repose sur le **Transfer Learning** (apprentissage par transfert) à l'aide de l'architecture **ResNet50**.
- **Pré-entraînement** : Le modèle est initialement pré-entraîné sur le vaste jeu de données ImageNet, ce qui lui permet de reconnaître de nombreuses caractéristiques visuelles (bords, textures, formes).
- **Adaptation** : Les couches extractrices de caractéristiques du modèle ResNet50 sont "gelées" (frozen). Seule la couche de classification finale (Fully Connected layer) est remplacée pour s'adapter spécifiquement à nos 38 classes.
- **Hyperparamètres** : Le modèle s'entraîne avec un batch size de 32, sur 10 époques avec l'optimiseur Adam et un learning rate de 0.001.

## 4. Améliorations Apportées
Le code original a été revu et optimisé sur les points suivants :
1. **Évaluation sur le set de Validation** : La boucle d'entraînement a été enrichie pour calculer et afficher non seulement les performances sur les données d'entraînement, mais aussi sur les données de validation à chaque époque.
2. **Sauvegarde du meilleur modèle (Model Checkpointing)** : Le code surveille désormais l'accuracy de validation. À la fin de l'entraînement, les poids du modèle restitués sont ceux qui ont obtenu la meilleure performance sur la validation, prévenant ainsi toute dégradation due à l'overfitting sur les dernières époques.
3. **Rapport de Classification** : Une analyse plus fine a été ajoutée. Lors de l'évaluation sur le test set, en plus de la matrice de confusion, un `classification_report` (Scikit-Learn) est généré pour fournir la précision, le rappel et le score F1 pour chacune des 38 classes.

## 5. Exécution et Résultats
- **Environnement** : Conçu pour s'exécuter dans un environnement type Google Colab (avec accélération GPU).
- **Déploiement** : Le modèle entraîné est exporté sous la forme du fichier `plant_resnet50_v1.pth` (poids du réseau) pour pouvoir être chargé ultérieurement dans une application web ou mobile d'inférence.
