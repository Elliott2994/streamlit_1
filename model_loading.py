import joblib  
  
def load_rf_model():  
    return joblib.load('rf_model.pkl')  
  
def load_gb_model():  
    return joblib.load('gb_model.pkl')
