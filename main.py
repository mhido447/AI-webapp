import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
import uvicorn
import os
import pickle
import numpy as np

app = FastAPI(title="Outil de Prédiction IA", version="1.0.0")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Models pour l'API
class MethodPredictionInput(BaseModel):
    nom_piece: str
    chapitre: str

class CostPredictionInput(BaseModel):
    methode_fabrication: str
    masse: float

class PredictionResponse(BaseModel):
    success: bool
    result: str
    error: str = None

# Variables globales pour les modèles et encodeurs
model_method = None
model_cost = None
label_encoder_piece = None
label_encoder_chapitre = None
label_encoder_methode = None

def load_models():
    """Charger et entraîner les modèles au démarrage"""
    global model_method, model_cost, label_encoder_piece, label_encoder_chapitre, label_encoder_methode
    
    try:
        # Chargement des données
        df = pd.read_excel("BDD 3 Finale.xlsx")
        
        # Encodage des variables catégorielles
        label_encoder_piece = LabelEncoder()
        df['nom_piece_encoded'] = label_encoder_piece.fit_transform(df['nom_piece'])

        label_encoder_chapitre = LabelEncoder()
        df['chapitre_encoded'] = label_encoder_chapitre.fit_transform(df['chapitre'])

        label_encoder_methode = LabelEncoder()
        df['methode_fabrication_encoded'] = label_encoder_methode.fit_transform(df['methode_fabrication'])

        # Modèle pour la prédiction de la méthode de fabrication
        X_method = df[['nom_piece_encoded', 'chapitre_encoded']]
        y_method = df['methode_fabrication_encoded']
        X_method_train, X_method_test, y_method_train, y_method_test = train_test_split(
            X_method, y_method, test_size=0.2, random_state=42
        )
        model_method = RandomForestClassifier(n_estimators=100, random_state=42)
        model_method.fit(X_method_train, y_method_train)

        # Modèle pour la prédiction du coût
        X_cost = df[['methode_fabrication_encoded', 'masse']]
        y_cost = df['coût']
        X_cost_train, X_cost_test, y_cost_train, y_cost_test = train_test_split(
            X_cost, y_cost, test_size=0.2, random_state=42
        )
        model_cost = RandomForestRegressor(n_estimators=100, random_state=42)
        model_cost.fit(X_cost_train, y_cost_train)

        print("✅ Modèles chargés avec succès")
        print("== Rapport sur la prédiction de méthode de fabrication ==")
        print(classification_report(y_method_test, model_method.predict(X_method_test), 
                                  target_names=label_encoder_methode.classes_))
        print("== Erreur quadratique moyenne sur la prédiction du coût ==")
        print(mean_squared_error(y_cost_test, model_cost.predict(X_cost_test)))
        
    except Exception as e:
        print(f"❌ Erreur lors du chargement des modèles: {e}")
        raise e

@app.on_event("startup")
async def startup_event():
    """Charger les modèles au démarrage de l'application"""
    load_models()

@app.get("/", response_class=HTMLResponse)
async def root():
    """Page principale"""
    return FileResponse('static/index.html')

@app.post("/predict-method", response_model=PredictionResponse)
async def predict_method(input_data: MethodPredictionInput):
    """Prédire la méthode de fabrication"""
    try:
        nom_piece_encoded = label_encoder_piece.transform([input_data.nom_piece])[0]
        chapitre_encoded = label_encoder_chapitre.transform([input_data.chapitre])[0]
        prediction_encoded = model_method.predict([[nom_piece_encoded, chapitre_encoded]])[0]
        prediction = label_encoder_methode.inverse_transform([prediction_encoded])[0]
        
        return PredictionResponse(
            success=True,
            result=f"Méthode prédite : {prediction}"
        )
    except ValueError as e:
        return PredictionResponse(
            success=False,
            result="",
            error="Nom de pièce ou chapitre inconnu dans la base de données"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-cost", response_model=PredictionResponse)
async def predict_cost(input_data: CostPredictionInput):
    """Prédire le coût"""
    try:
        methode_encoded = label_encoder_methode.transform([input_data.methode_fabrication])[0]
        prediction = model_cost.predict([[methode_encoded, input_data.masse]])[0]
        
        return PredictionResponse(
            success=True,
            result=f"Coût prédit : {prediction:.2f} €"
        )
    except ValueError as e:
        return PredictionResponse(
            success=False,
            result="",
            error="Méthode de fabrication inconnue dans la base de données"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/available-options")
async def get_available_options():
    """Obtenir les options disponibles pour les champs"""
    try:
        return {
            "pieces": label_encoder_piece.classes_.tolist(),
            "chapitres": label_encoder_chapitre.classes_.tolist(),
            "methodes": label_encoder_methode.classes_.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "models_loaded": model_method is not None and model_cost is not None
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)