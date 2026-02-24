import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

def generate_synthetic_data(n_samples=15000):
    np.random.seed(42)
    timestamp = [datetime(2025, 1, 1) + timedelta(minutes=10*i) for i in range(n_samples)]
    
    # Sensor data (Normal Distribution)
    voltage = np.random.normal(220, 5, n_samples)
    current = np.random.normal(10, 2, n_samples)
    frequency = np.random.normal(50, 0.5, n_samples)
    temperature = np.random.normal(45, 10, n_samples)
    network_traffic = np.random.poisson(100, n_samples)
    cpu_usage = np.random.uniform(20, 60, n_samples)
    
    # Cyclical/Time-dependent anomalies (e.g., Daytime peak)
    hours = np.array([t.hour for t in timestamp])
    current = np.where((hours > 8) & (hours < 18), current * 1.5, current)
    temperature = np.where((hours > 12) & (hours < 16), temperature + 15, temperature)
    
    df = pd.DataFrame({
        'Time': timestamp,
        'Voltage_V': voltage,
        'Current_A': current,
        'Frequency_Hz': frequency,
        'Temperature_C': temperature,
        'Network_Traffic_MBps': network_traffic,
        'CPU_Load_Percentage': cpu_usage
    })
    
    # Labeling: 0 = Normal, 1 = Hardware Failure, 2 = Cyber Attack (DDoS/Network Anomaly)
    status = np.zeros(n_samples, dtype=int)
    
    # Scenario 1: Hardware Failure (High temp, unstable freq, low voltage)
    failure_indices = np.random.choice(n_samples, int(n_samples * 0.10), replace=False)
    df.loc[failure_indices, 'Voltage_V'] -= np.random.uniform(20, 50, len(failure_indices))
    df.loc[failure_indices, 'Frequency_Hz'] -= np.random.uniform(1, 3, len(failure_indices))
    df.loc[failure_indices, 'Temperature_C'] += np.random.uniform(30, 60, len(failure_indices))
    status[failure_indices] = 1
    
    # Scenario 2: Cyber Attack (High network traffic, high CPU, normal electrical values)
    attack_indices = np.random.choice(n_samples, int(n_samples * 0.05), replace=False)
    # To avoid intersection
    attack_indices = np.setdiff1d(attack_indices, failure_indices)
    df.loc[attack_indices, 'Network_Traffic_MBps'] *= np.random.uniform(10, 50, len(attack_indices))
    df.loc[attack_indices, 'CPU_Load_Percentage'] += np.random.uniform(30, 40, len(attack_indices))
    df.loc[attack_indices, 'CPU_Load_Percentage'] = np.clip(df.loc[attack_indices, 'CPU_Load_Percentage'], 0, 100)
    status[attack_indices] = 2
    
    df['Target_Status'] = status
    return df

def feature_engineering(df):
    df_feat = df.copy()
    
    # Time series features
    df_feat['Hour'] = df_feat['Time'].dt.hour
    df_feat['Day_of_Week'] = df_feat['Time'].dt.dayofweek
    df_feat['Weekend'] = df_feat['Day_of_Week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Statistical / Interaction features
    df_feat['Approximate_Power_W'] = df_feat['Voltage_V'] * df_feat['Current_A']
    df_feat['Temperature_Current_Ratio'] = df_feat['Temperature_C'] / (df_feat['Current_A'] + 1e-5)
    df_feat['Network_CPU_Interaction'] = df_feat['Network_Traffic_MBps'] * df_feat['CPU_Load_Percentage']
    
    # Moving Averages (To reduce noise and capture trends)
    df_feat = df_feat.sort_values('Time').reset_index(drop=True)
    df_feat['Voltage_Moving_Avg_5'] = df_feat['Voltage_V'].rolling(window=5, min_periods=1).mean()
    df_feat['CPU_Moving_Avg_5'] = df_feat['CPU_Load_Percentage'].rolling(window=5, min_periods=1).mean()
    
    # Deviation Features
    df_feat['Voltage_Deviation'] = df_feat['Voltage_V'] - df_feat['Voltage_Moving_Avg_5']
    df_feat['CPU_Deviation'] = df_feat['CPU_Load_Percentage'] - df_feat['CPU_Moving_Avg_5']
    
    df_feat.drop('Time', axis=1, inplace=True)
    return df_feat

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    
    # Complex machine learning pipeline design
    pipeline_rf = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95, random_state=42)), # Dimensionality reduction
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    
    pipeline_gb = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42))
    ])
    
    models = {
        'Random Forest + PCA': pipeline_rf,
        'Gradient Boosting': pipeline_gb
    }
    
    best_model_name = ""
    best_f1 = 0
    best_predictions = None
    best_model = None
    
    print("="*70)
    print("🚀 Training Smart Grid Threat Analysis Models...")
    print("="*70)
    
    for name, model in models.items():
        print(f"\n[+] Training {name} model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"    - Accuracy: {acc:.4f}")
        print(f"    - F1-Score (Weighted): {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_predictions = y_pred
            best_model = model

    print("\n" + "="*70)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print("="*70)
    
    target_names = ['Normal Operation', 'Hardware Failure', 'Cyber Attack (DDoS)']
    
    print("\n📊 DETAILED CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_test, best_predictions, target_names=target_names))
    
    print("\n🧩 CONFUSION MATRIX")
    print("-" * 60)
    cm = confusion_matrix(y_test, best_predictions)
    cm_df = pd.DataFrame(cm, index=[f"True {i}" for i in target_names], columns=[f"Predicted {i}" for i in target_names])
    print(cm_df.to_string())
    
    # Feature Importances (for non-PCA model)
    if 'Gradient' in best_model_name:
        importances = best_model.named_steps['classifier'].feature_importances_
        feature_names = X_train.columns
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False).head(5)
        
        print("\n📈 TOP 5 CRITICAL FEATURES (Decision Mechanism)")
        print("-" * 60)
        for idx, row in importance_df.iterrows():
            print(f"  * {row['Feature']:<25}: {row['Importance']*100:.2f}%")

def main():
    print("1. Generating Synthetic Industrial IoT Data... (15,000 Records)")
    raw_data = generate_synthetic_data(n_samples=15000)
    
    print("2. Applying Feature Engineering...")
    processed_data = feature_engineering(raw_data)
    
    X = processed_data.drop('Target_Status', axis=1)
    y = processed_data['Target_Status']
    
    print("3. Splitting Dataset into Train and Test... (80% Train, 20% Test)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("   - Training Set Size:", X_train.shape)
    print("   - Test Set Size:  ", X_test.shape)
    print("   - Class Distribution:    \n", y_train.value_counts().to_string())
    
    train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    print("\n✅ Process Successfully Completed. Ready for LinkedIn Sharing.")

if __name__ == "__main__":
    main()