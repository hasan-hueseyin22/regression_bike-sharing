# Bike Sharing Demand Regression with CatBoost & Optuna

## 📝 Projektbeschreibung

Dieses Projekt befasst sich mit der Vorhersage der stündlichen Nachfrage nach Leihfahrrädern in einem Bike-Sharing-System. Ziel ist es, ein präzises Regressionsmodell zu entwickeln, das auf Basis von Wetter- und Zeitdaten die Anzahl der ausgeliehenen Fahrräder (`cnt`) prognostiziert.

Ein solches Modell kann Betreibern helfen, die Verfügbarkeit von Fahrrädern zu optimieren, Wartungsarbeiten zu planen und operative Entscheidungen zu treffen.

**Besonderheiten des Projekts:**
-   Einsatz von **CatBoost**, einem modernen und leistungsstarken Gradient-Boosting-Framework.
-   Automatische **Hyperparameter-Optimierung** mit **Optuna** zur Maximierung der Modellgenauigkeit.
-   Verarbeitung von Features mit zeitlichem Bezug und kategorialen Daten.

**Datensatz:** [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset) von der UCI Machine Learning Repository.

## 🛠️ Tech Stack

-   **Python**
-   **Pandas** für die Datenmanipulation
-   **Scikit-learn** für die Datenaufteilung und Metriken
-   **CatBoost** als Regressionsmodell
-   **Optuna** für die Hyperparameter-Optimierung
-   **Joblib** zum Speichern des trainierten Modells

## 🚀 Installation und Ausführung

1.  **Repository klonen:**
    ```bash
    git clone [https://github.com/hasan-hueseyin22/regression_bike-sharing.git](https://github.com/hasan-hueseyin22/regression_bike-sharing.git)
    cd bike-sharing-regression
    ```

2.  **Virtuelle Umgebung erstellen und aktivieren (empfohlen):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Auf Windows: venv\Scripts\activate
    ```

3.  **Abhängigkeiten installieren:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Modell trainieren und optimieren:**
    Das Skript startet den gesamten Prozess: Daten-Download, Hyperparameter-Suche mit Optuna und Training des finalen Modells.
    ```bash
    python src/train.py
    ```
    *Hinweis: Die Optuna-Optimierung kann je nach `OPTUNA_TRIALS`-Einstellung einige Minuten dauern.*

## 📊 Ergebnisse

Nach Abschluss der Optimierung werden die besten gefundenen Hyperparameter ausgegeben. Anschließend wird das finale Modell mit diesen Parametern trainiert und auf dem Test-Set evaluiert. Die finalen Metriken (**RMSE** und **R²-Score**) werden in der Konsole angezeigt. Das trainierte Modell wird als `models/catboost_model.joblib` gespeichert.

## 📂 Repository-Struktur
```
regression_bike-sharing/
├── data/
├── models/
├── notebooks/
├── src/
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```
