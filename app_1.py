import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import joblib
import os
import warnings
import networkx as nx  # Added for correlation network visualization

warnings.filterwarnings('ignore')

class SupplyChainDiagnostics:
    def __init__(self, model_dir='models'):
        """Initialize the supply chain diagnostics system"""
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Initialize models
        self.scaler = None
        self.pca = None
        self.iso_model = None
        self.db_model = None
        self.rf_model = None
        self.var_model = None
        
        # For tracking performance over time
        self.performance_history = {
            'timestamp': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Store causal relationships
        self.causal_matrix = {}
        
    def load_or_generate_data(self, filepath='supply_chain_data.csv', n_samples=1000, force_new=False):
        """Load existing data or generate new sample data"""
        if os.path.exists(filepath) and not force_new:
            print(f"Loading existing data from {filepath}")
            self.data = pd.read_csv(filepath)
        else:
            print("Generating new sample data...")
            self.data = self.generate_sample_data(n_samples)
            self.data.to_csv(filepath, index=False)
        
        print(f"Dataset loaded with {len(self.data)} records")
        return self.data
    
    def generate_sample_data(self, n_samples=1000):
        """
        Generate synthetic supply chain data for FMCG company
        """
        np.random.seed(42)

        # Date range for the last year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]

        # Randomly select dates (with replacement to simulate multiple events per day)
        selected_dates = np.random.choice(dates, n_samples)
        selected_dates = [date.strftime('%Y-%m-%d') for date in selected_dates]

        # Product catalog with linked categories and products
        product_catalog = [
            {'main_category': 'Beverages', 'product_name': 'Sparkling Citrus Soda'},
            {'main_category': 'Beverages', 'product_name': 'Mountain Spring Water'},
            {'main_category': 'Beverages', 'product_name': 'PowerCharge Berry Blast'},
            {'main_category': 'Personal Care', 'product_name': 'Smooth & Silky Shampoo'},
            {'main_category': 'Personal Care', 'product_name': 'MintFresh Toothpaste'},
            {'main_category': 'Personal Care', 'product_name': 'Coconut Oil Body Lotion'},
            {'main_category': 'Home Care', 'product_name': 'FreshLinen Laundry Detergent'},
            {'main_category': 'Home Care', 'product_name': 'Stain Remover Spray'},
            {'main_category': 'Home Care', 'product_name': 'Antibacterial Floor Cleaner'},
            {'main_category': 'Food', 'product_name': 'Tomato Paste'},
            {'main_category': 'Food', 'product_name': 'Whole Grain Bread'},
            {'main_category': 'Food', 'product_name': 'Instant Noodles'},
            {'main_category': 'Snacks', 'product_name': 'Sea Salt Potato Chips'},
            {'main_category': 'Snacks', 'product_name': 'Dark Chocolate Bar'},
            {'main_category': 'Snacks', 'product_name': 'Caramel Popcorn'},
            {'main_category': 'Baby Products', 'product_name': 'UltraDry Diapers'},
            {'main_category': 'Baby Products', 'product_name': 'Baby Wipes'},
            {'main_category': 'Baby Products', 'product_name': 'Baby Powder'},
            {'main_category': 'Pet Care', 'product_name': 'Dog Food'},
            {'main_category': 'Pet Care', 'product_name': 'Dog Chew Toys'},
            {'main_category': 'Pet Care', 'product_name': 'Grain-Free Dog Treats'}
        ]

        # Generate product selections
        selected_products = np.random.choice(product_catalog, n_samples)
        
        # Extract components
        main_categories = [p['main_category'] for p in selected_products]
        product_names = [p['product_name'] for p in selected_products]
        
        # Regions
        regions = ['North', 'South', 'East', 'West', 'Central']
        selected_regions = np.random.choice(regions, n_samples)

        # Suppliers mapping (expanded)
        suppliers_mapping = {
            'Beverages': ['BeverageCo Distributors', 'AquaPure Sources', 'EnergyFuel Inc.', 'FreshJuice Packers'],
            'Personal Care': ['PersonalCare Manufacturers', 'HygienePro Ltd.', 'Fresh&Clean Co.', 'Grooming Essentials Inc.'],
            'Home Care': ['CleanHome Solutions', 'EcoClean Suppliers', 'SparkleHouse Ltd.', 'ShineSmart Products'],
            'Food': ['FreshFoods International', 'PantryGoods Inc.', 'QualityConserve Foods', 'GoldenHarvest Ltd.'],
            'Snacks': ['CrunchTime Snacks', 'Sweet&Salty Ltd.', 'SnackMasters Co.', 'TastyBites International'],
            'Baby Products': ['TenderCare Baby', 'PureBaby Essentials', 'SafeNest Infant Care', 'LittleOnes Ltd.'],
            'Pet Care': ['PetHealth Distributors', 'HappyPaws Supplies', 'FurryFriends Ltd.', 'PremiumPet Care']
        }

        # Select suppliers based on main category
        selected_suppliers = [
            np.random.choice(suppliers_mapping[main_cat])
            for main_cat in main_categories
        ]

        # Normal operational metrics
        order_quantities = np.random.normal(5000, 1000, n_samples)
        lead_times = np.random.normal(5, 1, n_samples)
        transport_times = np.random.normal(3, 0.5, n_samples)
        inventory_levels = np.random.normal(8000, 1500, n_samples)
        demand_forecast = np.random.normal(4800, 900, n_samples)
        production_capacity = np.random.normal(6000, 500, n_samples)
        
        # New metrics for enhanced analysis
        supplier_performance = np.random.normal(85, 10, n_samples)  # Supplier performance score (0-100)
        quality_score = np.random.normal(95, 3, n_samples)  # Product quality score (0-100)
        market_growth = np.random.normal(2, 1, n_samples)  # Market growth percentage
        competitor_activity = np.random.normal(50, 15, n_samples)  # Competitor activity score (0-100)
        worker_availability = np.random.normal(90, 5, n_samples)  # Worker availability percentage

        # Introduce seasonal patterns
        seasonal_factor = np.sin(np.linspace(0, 2*np.pi, 365))
        seasonal_indices = [dates.index(datetime.strptime(date, '%Y-%m-%d').date())
                            if datetime.strptime(date, '%Y-%m-%d').date() in dates
                            else 0 for date in selected_dates]

        seasonal_effect = [seasonal_factor[i % len(seasonal_factor)] for i in seasonal_indices]

        # Ensure seasonal_effect has the same length as demand_forecast
        seasonal_effect = np.array(seasonal_effect)[:n_samples]

        demand_forecast = demand_forecast + seasonal_effect * 500

        # Introduce some anomalies (about 5% of the data)
        anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)

        # Supply disruption anomalies
        for idx in anomaly_indices[:len(anomaly_indices)//5]:
            lead_times[idx] *= np.random.uniform(2, 3)  # Significantly longer lead times
            inventory_levels[idx] *= np.random.uniform(0.2, 0.5)  # Much lower inventory
            supplier_performance[idx] *= np.random.uniform(0.5, 0.7)  # Supplier performance issues

        # Demand spike anomalies
        for idx in anomaly_indices[len(anomaly_indices)//5:2*len(anomaly_indices)//5]:
            demand_forecast[idx] *= np.random.uniform(1.5, 2.5)  # Unexpected demand spikes
            inventory_levels[idx] *= np.random.uniform(0.3, 0.6)  # Lower inventory due to spikes
            market_growth[idx] *= np.random.uniform(2, 3)  # Market growth spikes

        # Production issues anomalies
        for idx in anomaly_indices[2*len(anomaly_indices)//5:3*len(anomaly_indices)//5]:
            production_capacity[idx] *= np.random.uniform(0.4, 0.7)  # Production capacity issues
            transport_times[idx] *= np.random.uniform(1.5, 2.5)  # Logistics delays
            worker_availability[idx] *= np.random.uniform(0.6, 0.8)  # Worker availability issues

        # Quality issues
        for idx in anomaly_indices[3*len(anomaly_indices)//5:4*len(anomaly_indices)//5]:
            quality_score[idx] *= np.random.uniform(0.7, 0.85)  # Quality issues
            order_quantities[idx] *= np.random.uniform(0.6, 0.8)  # Reduced orders due to quality

        # Competitor disruptions
        for idx in anomaly_indices[4*len(anomaly_indices)//5:]:
            competitor_activity[idx] *= np.random.uniform(1.5, 2)  # Increased competitor activity
            demand_forecast[idx] *= np.random.uniform(0.7, 0.9)  # Reduced demand due to competition
            market_growth[idx] *= np.random.uniform(0.5, 0.8)  # Reduced market growth

        # Weather impact (random severe weather days)
        weather_impact = np.zeros(n_samples)
        severe_weather_days = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
        weather_impact[severe_weather_days] = np.random.uniform(0.5, 1.0, size=len(severe_weather_days))

        # Political stability (regional issues)
        political_stability = np.ones(n_samples) * 90  # Base political stability score (0-100)
        unstable_regions = {
            'East': np.random.choice(np.where(np.array(selected_regions) == 'East')[0], 
                                    size=int(len(np.where(np.array(selected_regions) == 'East')[0]) * 0.2)),
            'South': np.random.choice(np.where(np.array(selected_regions) == 'South')[0], 
                                     size=int(len(np.where(np.array(selected_regions) == 'South')[0]) * 0.1))
        }
        
        for region, indices in unstable_regions.items():
            political_stability[indices] *= np.random.uniform(0.6, 0.8, size=len(indices))
            # Also affect transport times in unstable regions
            transport_times[indices] *= np.random.uniform(1.3, 1.8, size=len(indices))

        # Create DataFrame
        df = pd.DataFrame({
            'date': selected_dates,
            'main_category': main_categories,
            'product_name': product_names,
            'region': selected_regions,
            'supplier': selected_suppliers,
            'order_quantity': order_quantities,
            'lead_time_days': lead_times,
            'transport_time_days': transport_times,
            'inventory_level': inventory_levels,
            'demand_forecast': demand_forecast,
            'production_capacity': production_capacity,
            'supplier_performance': supplier_performance,
            'quality_score': quality_score,
            'market_growth': market_growth,
            'competitor_activity': competitor_activity,
            'worker_availability': worker_availability,
            'weather_impact': weather_impact,
            'political_stability': political_stability,
            # Derived metrics
            'inventory_coverage_days': inventory_levels / (demand_forecast / 30),
            'capacity_utilization': demand_forecast / production_capacity,
            'supply_demand_ratio': inventory_levels / demand_forecast,
        })

        # Add a few extreme outliers
        extreme_indices = np.random.choice(n_samples, size=10, replace=False)
        df.loc[extreme_indices, 'lead_time_days'] *= 5
        df.loc[extreme_indices, 'transport_time_days'] *= 4

        # Label the true anomalies for evaluation
        df['true_anomaly'] = 0
        df.loc[anomaly_indices, 'true_anomaly'] = 1
        df.loc[extreme_indices, 'true_anomaly'] = 1
        df.loc[severe_weather_days, 'true_anomaly'] = 1
        
        # Add anomaly type labels for training causal models
        df['anomaly_type'] = 'normal'
        df.loc[anomaly_indices[:len(anomaly_indices)//5], 'anomaly_type'] = 'supply_disruption'
        df.loc[anomaly_indices[len(anomaly_indices)//5:2*len(anomaly_indices)//5], 'anomaly_type'] = 'demand_spike'
        df.loc[anomaly_indices[2*len(anomaly_indices)//5:3*len(anomaly_indices)//5], 'anomaly_type'] = 'production_issue'
        df.loc[anomaly_indices[3*len(anomaly_indices)//5:4*len(anomaly_indices)//5], 'anomaly_type'] = 'quality_issue'
        df.loc[anomaly_indices[4*len(anomaly_indices)//5:], 'anomaly_type'] = 'competitor_disruption'
        df.loc[severe_weather_days, 'anomaly_type'] = 'weather_impact'
        
        # Shuffle the dataframe
        return df.sample(frac=1, random_state=42).reset_index(drop=True)

    def preprocess_data(self, df=None):
        """
        Preprocess the supply chain data for anomaly detection
        """
        if df is None:
            df = self.data
            
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])

        # One-hot encode categorical variables
        df_processed = pd.get_dummies(df, columns=['main_category','product_name', 'region', 'supplier'])

        # Extract features for anomaly detection
        self.features = [
            'order_quantity', 'lead_time_days', 'transport_time_days',
            'inventory_level', 'demand_forecast', 'production_capacity',
            'weather_impact', 'inventory_coverage_days', 'capacity_utilization'
        ]

        # Scale the features
        self.scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            self.scaler.fit_transform(df_processed[self.features]),
            columns=self.features
        )
        
        # Save scaler for future use
        joblib.dump(self.scaler, f"{self.model_dir}/scaler.pkl")

        return df_scaled, self.features, df_processed
    
    def build_models(self, X_scaled):
        """
        Build and train all the required models
        """
        # 1. Isolation Forest for anomaly detection
        print("Training Isolation Forest model...")
        self.iso_model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=0.05,
            random_state=42
        )
        self.iso_model.fit(X_scaled)
        joblib.dump(self.iso_model, f"{self.model_dir}/isolation_forest.pkl")
        
        # 2. PCA for dimensionality reduction
        print("Performing PCA for dimensionality reduction...")
        self.pca = PCA(n_components=0.95)  # Retain 95% of variance
        X_pca = self.pca.fit_transform(X_scaled)
        joblib.dump(self.pca, f"{self.model_dir}/pca.pkl")
        
        # 3. DBSCAN for cluster-based anomaly detection
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(X_pca)
        distances, _ = nn.kneighbors(X_pca)
        distances = np.sort(distances[:, 1])
        knee_point = np.argmax(distances[1:] - distances[:-1]) + 1
        eps = distances[knee_point]
        
        print(f"Training DBSCAN with eps={eps:.4f}...")
        self.db_model = DBSCAN(eps=eps, min_samples=5)
        self.db_model.fit(X_pca)
        joblib.dump(self.db_model, f"{self.model_dir}/dbscan.pkl")
        
        return self.iso_model, self.pca, self.db_model
    
    def detect_anomalies(self, X_scaled=None):
        """
        Detect anomalies using ensemble method
        """
        if X_scaled is None:
            X_scaled, _, _ = self.preprocess_data()
            
        # Apply Isolation Forest
        iso_pred = self.iso_model.predict(X_scaled)
        isolation_anomalies = np.where(iso_pred == -1, 1, 0)
        
        # Apply PCA + DBSCAN
        X_pca = self.pca.transform(X_scaled)
        clusters = self.db_model.fit_predict(X_pca)
        dbscan_anomalies = np.where(clusters == -1, 1, 0)
        
        # Combine methods (a point is anomalous if either method flags it)
        ensemble_anomalies = np.logical_or(isolation_anomalies, dbscan_anomalies).astype(int)
        
        anomaly_results = {
            'isolation_forest': isolation_anomalies,
            'dbscan': dbscan_anomalies,
            'ensemble': ensemble_anomalies
        }
        
        # Store detected anomaly indices
        self.anomaly_indices = np.where(anomaly_results['ensemble'] == 1)[0]
        
        return anomaly_results
    
    def build_causal_model(self, data=None):
        """
        Build a causal model to identify relationships between variables
        """
        if data is None:
            data = self.data
            
        print("Building causal inference model...")
        # Prepare time series data for causal analysis
        data_ts = data.sort_values('date').copy()
        
        # Group by date and aggregate metrics
        daily_data = data_ts.groupby('date')[self.features].mean().reset_index()
        daily_data.set_index('date', inplace=True)
        
        # Ensure the time series is stationary for Granger causality test
        # Compute first differences
        diff_data = daily_data.diff().dropna()
        
        # Test for Granger causality between all pairs of variables
        self.causal_matrix = {}
        
        # We'll test each variable as a potential cause for each other variable
        # Focus on key metrics for disruption
        target_vars = ['lead_time_days', 'inventory_coverage_days', 
                       'transport_time_days', 'capacity_utilization']
        
        causal_vars = [var for var in self.features if var not in target_vars]
        
        print("Testing Granger causality between variables...")
        for target in target_vars:
            self.causal_matrix[target] = {}
            for causal in causal_vars:
                if causal != target:  # Don't test variable against itself
                    # Prepare data
                    test_data = diff_data[[target, causal]].dropna()
                    
                    if len(test_data) > 10:  # Need sufficient data for testing
                        try:
                            # Test Granger causality from causal -> target
                            test_result = grangercausalitytests(test_data[[target, causal]], 
                                                              maxlag=5, verbose=False)
                            
                            # Extract p-values for each lag
                            p_values = [test_result[i+1][0]['ssr_chi2test'][1] for i in range(5)]
                            min_p_value = min(p_values)
                            
                            # Store results for significant relationships (p < 0.05)
                            if min_p_value < 0.05:
                                self.causal_matrix[target][causal] = {
                                    'p_value': min_p_value,
                                    'causal_impact': 'strong' if min_p_value < 0.01 else 'moderate'
                                }
                        except:
                            # Skip if there are numerical issues
                            pass
        
        # Build VAR model for time series analysis and forecasting
        print("Building VAR model for time series forecasting...")
        try:
            model = VAR(diff_data)
            self.var_model = model.fit(maxlags=5, ic='aic')
            joblib.dump(self.var_model, f"{self.model_dir}/var_model.pkl")
        except:
            print("Warning: VAR model could not be fitted. Proceeding without it.")
            self.var_model = None
        
        return self.causal_matrix
    
    def build_classification_model(self):
        """
        Build a classification model to identify anomaly types
        """
        print("Building classification model for anomaly type prediction...")
        # Filter to only include known anomalies for training
        anomaly_data = self.data[self.data['true_anomaly'] == 1].copy()
        
        if len(anomaly_data) > 20:  # Need sufficient data for training
            # Prepare features and target
            available_features = [f for f in self.features if f in anomaly_data.columns]
            X = anomaly_data[available_features]
            y = anomaly_data['anomaly_type']
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, random_state=42
            )
            
            # Train Random Forest classifier
            self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.rf_model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.rf_model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"Classification model performance:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            # Save model
            joblib.dump(self.rf_model, f"{self.model_dir}/rf_classifier.pkl")
            
            # Update performance history
            self.performance_history['timestamp'].append(datetime.now())
            self.performance_history['accuracy'].append(accuracy)
            self.performance_history['precision'].append(precision)
            self.performance_history['recall'].append(recall)
            self.performance_history['f1'].append(f1)
            
            return self.rf_model
        else:
            print("Not enough anomaly data to build classification model.")
            return None
    
    def identify_root_causes(self, anomaly_indices=None, data=None):
        """
        Identify potential root causes for detected anomalies
        using multi-method approach
        """
        if data is None:
            data = self.data
        
        if anomaly_indices is None:
            anomaly_indices = self.anomaly_indices
            
        anomalous_data = data.iloc[anomaly_indices]
        normal_data = data.iloc[~np.isin(np.arange(len(data)), anomaly_indices)]

        root_causes = {}
        
        # 1. Statistical Approach - Z-score analysis
        print("Performing statistical root cause analysis...")
        means = normal_data[self.features].mean()
        stds = normal_data[self.features].std()

        z_scores = (anomalous_data[self.features] - means) / stds

        # 2. Classification Approach - Predict anomaly type
        anomaly_types = {}
        if self.rf_model is not None:
            print("Classifying anomaly types...")
            X_anomaly = self.scaler.transform(anomalous_data[self.features])
            predicted_types = self.rf_model.predict(X_anomaly)
            
            for i, idx in enumerate(anomaly_indices):
                anomaly_types[idx] = predicted_types[i]
        
        # 3. Causal Inference - Use Granger causality results
        print("Applying causal inference for root cause analysis...")
        
        # Combine all approaches
        for i, idx in enumerate(anomaly_indices):
            significant_dev = []
            
            # Get statistical deviations
            row = z_scores.iloc[i]
            for feature, value in row.items():
                if abs(value) > 2:  # Z-score threshold
                    direction = "high" if value > 0 else "low"
                    impact = "critical" if abs(value) > 4 else "significant" if abs(value) > 3 else "moderate"
                    significant_dev.append({
                        'feature': feature, 
                        'z_score': value, 
                        'direction': direction,
                        'impact': impact,
                        'analysis_type': 'statistical'
                    })
            
            # Add anomaly type if available
            if self.rf_model is not None and idx in anomaly_types:
                predicted_type = anomaly_types[idx]
                significant_dev.append({
                    'feature': 'anomaly_classification',
                    'value': predicted_type,
                    'analysis_type': 'classification'
                })
                
                # Add causal factors based on classification and causal matrix
                if predicted_type == 'supply_disruption':
                    target_var = 'lead_time_days'
                elif predicted_type == 'demand_spike':
                    target_var = 'inventory_coverage_days'
                elif predicted_type == 'production_issue':
                    target_var = 'capacity_utilization'
                elif predicted_type == 'weather_impact' or predicted_type == 'competitor_disruption':
                    target_var = 'transport_time_days'
                else:
                    target_var = None
                    
                if target_var and target_var in self.causal_matrix:
                    for causal_var, info in self.causal_matrix[target_var].items():
                        if info['p_value'] < 0.05:
                            # Check if this anomaly has an extreme value for this causal variable
                            z_value = row.get(causal_var)
                            if z_value and abs(z_value) > 1.5:  # More relaxed threshold for causal variables
                                significant_dev.append({
                                    'feature': causal_var,
                                    'z_score': z_value,
                                    'direction': 'high' if z_value > 0 else 'low',
                                    'causal_impact': info['causal_impact'],
                                    'p_value': info['p_value'],
                                    'analysis_type': 'causal'
                                })
            
            # Sort by absolute z-score and analysis type (prioritize causal factors)
            significant_dev.sort(key=lambda x: 
                               (0 if x.get('analysis_type') == 'causal' else 
                                1 if x.get('analysis_type') == 'classification' else 2,
                                -abs(x.get('z_score', 0)) if 'z_score' in x else 0))
            
            root_causes[idx] = significant_dev
        
        return root_causes
    
    def generate_recommendations(self, root_causes, data=None):
        """
        Generate actionable recommendations based on root causes
        """
        if data is None:
            data = self.data
            
        recommendations = {}
        
        for idx, causes in root_causes.items():
            rec_list = []
            anomaly_record = data.iloc[idx]
            
            # Extract key information
            anomaly_type = None
            for cause in causes:
                if cause.get('feature') == 'anomaly_classification':
                    anomaly_type = cause.get('value')
                    break
            
            # Get category, product, region and supplier
            category = anomaly_record['main_category']
            product = anomaly_record['product_name']
            region = anomaly_record['region']
            supplier = anomaly_record['supplier']
            
            # Base recommendations on anomaly type and root causes
            if anomaly_type == 'supply_disruption':
                rec_list.append({
                    'priority': 'high',
                    'action': f'Initiate supplier review with {supplier}',
                    'details': 'Schedule meeting to address performance issues and establish improvement plan'
                })
                rec_list.append({
                    'priority': 'medium',
                    'action': 'Increase safety stock levels',
                    'details': f'For {product} in {region} region, increase buffer inventory by 30%'
                })
                rec_list.append({
                    'priority': 'medium',
                    'action': 'Activate alternate supplier',
                    'details': f'Temporarily source {category} products from backup suppliers'
                })
                
            elif anomaly_type == 'demand_spike':
                rec_list.append({
                    'priority': 'high',
                    'action': 'Expedite production and shipping',
                    'details': f'For {product}, authorize overtime and premium freight'
                })
                rec_list.append({
                    'priority': 'medium',
                    'action': 'Adjust demand forecast',
                    'details': f'Update {region} forecast for {product} upward by 25%'
                })
                rec_list.append({
                    'priority': 'medium',
                    'action': 'Investigate market drivers',
                    'details': f'Conduct market analysis to determine if spike is temporary or sustained'
                })
                
            elif anomaly_type == 'production_issue':
                rec_list.append({
                    'priority': 'high',
                    'action': 'Schedule urgent maintenance',
                    'details': 'Allocate emergency maintenance team to resolve capacity constraints'
                })
                rec_list.append({
                    'priority': 'medium',
                    'action': 'Reallocate production capacity',
                    'details': f'Shift production of {product} to alternate facilities temporarily'
                })
                rec_list.append({
                    'priority': 'low',
                    'action': 'Review maintenance protocols',
                    'details': 'Update preventative maintenance schedule to prevent recurrence'
                })
                
            elif anomaly_type == 'quality_issue':
                rec_list.append({
                    'priority': 'high',
                    'action': 'Conduct quality audit',
                    'details': f'Detailed inspection of {product} from {supplier}'
                })
                rec_list.append({
                    'priority': 'high',
                    'action': 'Hold shipments',
                    'details': f'Temporarily suspend distribution of affected batches until quality verification'
                })
                rec_list.append({
                    'priority': 'medium',
                    'action': 'Supplier quality review',
                    'details': f'Schedule quality improvement discussion with {supplier}'
                })
                
            elif anomaly_type == 'competitor_disruption':
                rec_list.append({
                    'priority': 'high',
                    'action': 'Competitive response strategy',
                    'details': f'Develop targeted promotion for {product} in {region} region'
                })
                rec_list.append({
                    'priority': 'medium',
                    'action': 'Market share analysis',
                    'details': 'Conduct detailed analysis of competitor activities and pricing'
                })
                rec_list.append({
                    'priority': 'medium',
                    'action': 'Adjust pricing strategy',
                    'details': f'Consider temporary price adjustments for {product}'
                })
                
            elif anomaly_type == 'weather_impact':
                rec_list.append({
                    'priority': 'high',
                    'action': 'Activate alternate logistics routes',
                    'details': f'Reroute shipments to {region} through alternative distribution channels'
                })
                rec_list.append({
                    'priority': 'medium',
                    'action': 'Deploy regional inventory',
                    'details': f'Transfer stock from neighboring regions to support {region}'
                })
                rec_list.append({
                    'priority': 'low',
                    'action': 'Update weather contingency plans',
                    'details': 'Review and enhance regional weather response procedures'
                })
                
            else:
                # Generic recommendations based on statistical factors
                for cause in causes:
                    if cause.get('analysis_type') == 'statistical':
                        feature = cause.get('feature')
                        direction = cause.get('direction')
                        
                        if feature == 'lead_time_days' and direction == 'high':
                            rec_list.append({
                                'priority': 'high',
                                'action': 'Investigate supplier delays',
                                'details': f'Contact {supplier} regarding extended lead times'
                            })
                        
                        elif feature == 'inventory_level' and direction == 'low':
                            rec_list.append({
                                'priority': 'high',
                                'action': 'Replenish inventory',
                                'details': f'Expedite orders for {product} to rebuild stock'
                            })
                        
                        elif feature == 'quality_score' and direction == 'low':
                            rec_list.append({
                                'priority': 'medium',
                                'action': 'Quality control review',
                                'details': f'Inspect recent {product} shipments for issues'
                            })
            
            # Add at least one recommendation if none were generated
            if not rec_list:
                rec_list.append({
                    'priority': 'medium',
                    'action': 'Investigate anomaly',
                    'details': f'Conduct detailed review of {product} in {region} region supplied by {supplier}'
                })
                
            recommendations[idx] = rec_list
        
        return recommendations

    def run_full_analysis(self, data=None):
        """
        Run a complete end-to-end analysis pipeline
        """
        if data is None:
            # Load or generate data
            data = self.load_or_generate_data()
        
        print("Starting full supply chain analysis...")
        
        # Preprocess data
        print("Preprocessing data...")
        X_scaled, features, processed_data = self.preprocess_data(data)
        
        # Build models
        print("Building models...")
        self.build_models(X_scaled)
        
        # Detect anomalies
        print("Detecting anomalies...")
        anomaly_results = self.detect_anomalies(X_scaled)
        
        # Build causal model
        print("Building causal model...")
        self.build_causal_model(data)
        
        # Build classification model
        print("Building classification model...")
        self.build_classification_model()
        
        # Identify root causes
        print("Identifying root causes...")
        root_causes = self.identify_root_causes(
            np.where(anomaly_results['ensemble'] == 1)[0], data
        )
        
        # Generate recommendations
        print("Generating recommendations...")
        recommendations = self.generate_recommendations(root_causes, data)
        
        # Return complete results
        results = {
            'anomalies_detected': sum(anomaly_results['ensemble']),
            'isolation_forest_anomalies': sum(anomaly_results['isolation_forest']),
            'dbscan_anomalies': sum(anomaly_results['dbscan']),
            'anomaly_indices': np.where(anomaly_results['ensemble'] == 1)[0].tolist(),
            'anomaly_records': data.iloc[np.where(anomaly_results['ensemble'] == 1)[0]].to_dict('records'),
            'root_causes': root_causes,
            'recommendations': recommendations,
            'causal_relationships': self.causal_matrix
        }
        
        print(f"Analysis complete. {results['anomalies_detected']} anomalies identified.")
        return results
    
    def launch_dashboard(self, data=None, port=8050):
        """
        Launch an interactive dashboard for exploring the analysis results
        """
        if data is None:
            data = self.data
            
        # Make a copy to avoid changing the original data
        dash_data = data.copy()
        
        # Ensure we have anomaly predictions
        if not hasattr(self, 'anomaly_indices'):
            X_scaled, _, _ = self.preprocess_data(dash_data)
            self.detect_anomalies(X_scaled)
            
        # Add anomaly predictions to the data
        dash_data['predicted_anomaly'] = 0
        dash_data.loc[self.anomaly_indices, 'predicted_anomaly'] = 1
        
        # Get root causes and recommendations
        if not hasattr(self, 'causal_matrix'):
            self.build_causal_model(dash_data)
            
        root_causes = self.identify_root_causes(self.anomaly_indices, dash_data)
        recommendations = self.generate_recommendations(root_causes, dash_data)
        
        # Convert datetime to string for easier handling in Dash
        if 'date' in dash_data.columns:
            dash_data['date'] = dash_data['date'].astype(str)
            
        # Create the Dash app
        app = dash.Dash(__name__, suppress_callback_exceptions=True)
        
        # Define app layout
        app.layout = html.Div([
            html.H1("Supply Chain Diagnostics Dashboard"),
            
            # Tabs for different views
            dcc.Tabs([
                # Overview Tab
                dcc.Tab(label="Overview", children=[
                    html.Div([
                        html.H3("Supply Chain Performance Overview"),
                        
                        # High-level metrics
                        html.Div([
                            html.Div([
                                html.H4("Total Records"),
                                html.H2(f"{len(dash_data)}")
                            ], className="metric-card"),
                            
                            html.Div([
                                html.H4("Anomalies Detected"),
                                html.H2(f"{len(self.anomaly_indices)}")
                            ], className="metric-card"),
                            
                            html.Div([
                                html.H4("Anomaly Rate"),
                                html.H2(f"{len(self.anomaly_indices)/len(dash_data)*100:.1f}%")
                            ], className="metric-card"),
                            
                            html.Div([
                                html.H4("Categories"),
                                html.H2(f"{dash_data['main_category'].nunique()}")
                            ], className="metric-card")
                        ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'space-between'}),
                        
                        # Category breakdown chart
                        html.Div([
                            html.H3("Anomaly Distribution by Category"),
                            dcc.Graph(
                                id='category-chart',
                                figure=px.bar(
                                    dash_data.groupby('main_category')['predicted_anomaly'].mean().reset_index(),
                                    x='main_category',
                                    y='predicted_anomaly',
                                    title="Anomaly Rate by Category",
                                    labels={'predicted_anomaly': 'Anomaly Rate', 'main_category': 'Category'}
                                )
                            )
                        ], style={'marginTop': '20px'}),
                        
                        # Time series view
                        html.Div([
                            html.H3("Anomaly Timeline"),
                            dcc.Graph(
                                id='timeline-chart',
                                figure=px.scatter(
                                    dash_data.sort_values('date'),
                                    x='date',
                                    y='lead_time_days',
                                    color='predicted_anomaly',
                                    color_discrete_map={0: 'blue', 1: 'red'},
                                    title="Lead Time Anomalies Over Time",
                                    labels={'lead_time_days': 'Lead Time (Days)', 'date': 'Date'}
                                )
                            )
                        ], style={'marginTop': '20px'})
                    ])
                ]),
                
                # Anomalies Tab
                dcc.Tab(label="Anomaly Details", children=[
                    html.Div([
                        html.H3("Detected Anomalies"),
                        
                        # Filter controls
                        html.Div([
                            html.Label("Filter by Category:"),
                            dcc.Dropdown(
                                id='category-filter',
                                options=[{'label': cat, 'value': cat} 
                                         for cat in dash_data['main_category'].unique()],
                                multi=True,
                                placeholder="Select categories..."
                            ),
                            
                            html.Label("Filter by Region:"),
                            dcc.Dropdown(
                                id='region-filter',
                                options=[{'label': region, 'value': region} 
                                         for region in dash_data['region'].unique()],
                                multi=True,
                                placeholder="Select regions..."
                            )
                        ]),
                        
                        # Anomalies data table
                        html.Div([
                            dash_table.DataTable(
                                id='anomalies-table',
                                columns=[
                                    {'name': 'Date', 'id': 'date'},
                                    {'name': 'Category', 'id': 'main_category'},
                                    {'name': 'Product', 'id': 'product_name'},
                                    {'name': 'Region', 'id': 'region'},
                                    {'name': 'Supplier', 'id': 'supplier'},
                                    {'name': 'Lead Time', 'id': 'lead_time_days'},
                                    {'name': 'Inventory', 'id': 'inventory_level'},
                                    {'name': 'Demand', 'id': 'demand_forecast'}
                                ],
                                data=dash_data[dash_data['predicted_anomaly'] == 1].to_dict('records'),
                                sort_action='native',
                                filter_action='native',
                                page_size=10,
                                style_data_conditional=[
                                    {
                                        'if': {'row_index': 'odd'},
                                        'backgroundColor': 'rgb(248, 248, 248)'
                                    }
                                ],
                                style_header={
                                    'backgroundColor': 'rgb(230, 230, 230)',
                                    'fontWeight': 'bold'
                                }
                            )
                        ], style={'marginTop': '20px'})
                    ])
                ]),
                
                # Root Causes Tab
                dcc.Tab(label="Root Cause Analysis", children=[
                    html.Div([
                        html.H3("Root Cause Analysis"),
                        
                        # Anomaly selector
                        html.Div([
                            html.Label("Select Anomaly:"),
                            dcc.Dropdown(
                                id='anomaly-selector',
                                options=[
                                    {'label': f"{dash_data.iloc[idx]['date']} - {dash_data.iloc[idx]['product_name']} ({dash_data.iloc[idx]['region']})", 
                                     'value': idx} 
                                    for idx in self.anomaly_indices
                                ],
                                placeholder="Select an anomaly to analyze..."
                            )
                        ]),
                        
                        # Root cause display
                        html.Div(id='root-cause-display'),
                        
                        # Recommendations display
                        html.Div(id='recommendations-display')
                    ])
                ]),
                
                # Causal Relationships Tab
                dcc.Tab(label="Causal Relationships", children=[
                    html.Div([
                        html.H3("Causal Relationship Analysis"),
                        
                        # Heatmap of causal relationships
                        html.Div([
                            html.H4("Variable Impact Heatmap"),
                            dcc.Graph(id='causal-heatmap')
                        ]),
                        
                        # Correlation network
                        html.Div([
                            html.H4("Feature Correlation Network"),
                            dcc.Graph(id='correlation-network')
                        ])
                    ])
                ])
            ])
        ])
        
        # Define callbacks
        @app.callback(
            [Output('root-cause-display', 'children'),
             Output('recommendations-display', 'children')],
            [Input('anomaly-selector', 'value')]
        )
        def update_root_cause(anomaly_idx):
            if anomaly_idx is None:
                return html.Div(), html.Div()
            
            # Display root causes
            anomaly_record = dash_data.iloc[anomaly_idx]
            causes = root_causes.get(anomaly_idx, [])
            
            causes_display = [
                html.H4(f"Root Causes for {anomaly_record['product_name']} ({anomaly_record['date']})"),
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Factor"),
                            html.Th("Analysis Type"),
                            html.Th("Impact"),
                            html.Th("Details")
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(cause.get('feature')),
                            html.Td(cause.get('analysis_type')),
                            html.Td(cause.get('impact', 'moderate')),
                            html.Td(f"{cause.get('direction', '')} (z-score: {cause.get('z_score', 'N/A'):.2f})" 
                                   if 'z_score' in cause else cause.get('value', ''))
                        ]) for cause in causes
                    ])
                ])
            ]
            
            # Display recommendations
            recs = recommendations.get(anomaly_idx, [])
            
            recs_display = [
                html.H4("Recommended Actions"),
                html.Table([
                    html.Thead(
                        html.Tr([
                            html.Th("Priority"),
                            html.Th("Action"),
                            html.Th("Details")
                        ])
                    ),
                    html.Tbody([
                        html.Tr([
                            html.Td(rec.get('priority', 'medium'), 
                                   style={'backgroundColor': 
                                         'red' if rec.get('priority') == 'high' else
                                         'orange' if rec.get('priority') == 'medium' else 'yellow'}),
                            html.Td(rec.get('action', '')),
                            html.Td(rec.get('details', ''))
                        ]) for rec in recs
                    ])
                ])
            ]
            
            return html.Div(causes_display), html.Div(recs_display)
        
        @app.callback(
            Output('causal-heatmap', 'figure'),
            [Input('causal-heatmap', 'id')]  # Dummy input to trigger callback on load
        )
        def update_causal_heatmap(_):
            # Convert causal matrix to dataframe for visualization
            causal_data = []
            
            for target, causes in self.causal_matrix.items():
                for cause, details in causes.items():
                    causal_data.append({
                        'target': target,
                        'cause': cause,
                        'p_value': details['p_value'],
                        'strength': -np.log10(details['p_value'])  # Transform for better visualization
                    })
            
            if not causal_data:
                # Return empty figure if no causal relationships
                return go.Figure()
            
            causal_df = pd.DataFrame(causal_data)
            
            # Create heatmap
            fig = px.density_heatmap(
                causal_df,
                x='cause',
                y='target',
                z='strength',
                title="Causal Relationship Strength",
                labels={'strength': 'Strength (-log10 p-value)'}
            )
            
            return fig
        
        @app.callback(
            Output('correlation-network', 'figure'),
            [Input('correlation-network', 'id')]  # Dummy input
        )
        def update_correlation_network(_):
            # Calculate correlation matrix
            corr_matrix = dash_data[self.features].corr()
            
            # Create network graph
            G = nx.Graph() if 'nx' in globals() else None
            
            if G is None:
                # Return empty figure if networkx not available
                return go.Figure()
            
            # Add nodes
            for feature in self.features:
                G.add_node(feature)
            
            # Add edges for strong correlations (positive or negative)
            for i, feat1 in enumerate(self.features):
                for j, feat2 in enumerate(self.features):
                    if i < j:  # Avoid duplicates
                        corr = corr_matrix.loc[feat1, feat2]
                        if abs(corr) > 0.5:  # Only strong correlations
                            G.add_edge(feat1, feat2, weight=abs(corr), sign=np.sign(corr))
            
            # Create network layout
            pos = nx.spring_layout(G)
            
            # Create plot
            edge_trace = []
            for u, v, data in G.edges(data=True):
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                weight = data['weight']
                sign = data['sign']
                
                color = 'blue' if sign > 0 else 'red'
                
                edge_trace.append(
                    go.Scatter(
                        x=[x0, x1, None],
                        y=[y0, y1, None],
                        line=dict(width=weight*3, color=color),
                        hoverinfo='none',
                        mode='lines'
                    )
                )
            
            # Create nodes
            node_x = []
            node_y = []
            node_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="top center",
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=15,
                    colorbar=dict(
                        thickness=15,
                        title='Node Connections',
                        xanchor='left',
                        titleside='right'
                    )
                )
            )
            
            # Color nodes by number of connections
            node_adjacencies = []
            for node, adjacencies in enumerate(G.adjacency()):
                node_adjacencies.append(len(adjacencies[1]))
            
            node_trace.marker.color = node_adjacencies
            
            # Create the figure
            fig = go.Figure(data=edge_trace + [node_trace],
                          layout=go.Layout(
                              title="Feature Correlation Network",
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20, l=5, r=5, t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                          ))
            
            return fig
        
        # Launch the dashboard
        print(f"Launching interactive dashboard on port {port}...")
        app.run(debug=True, port=port)
        
        return app

# Usage example
if __name__ == "__main__":
    # Initialize the supply chain diagnostics system
    diagnostics = SupplyChainDiagnostics()
    
    # Run full analysis
    results = diagnostics.run_full_analysis()
    
    # Launch interactive dashboard
    diagnostics.launch_dashboard()