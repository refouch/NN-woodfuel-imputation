import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
# Import pour la gestion des distances
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances, pairwise_distances
from typing import List


def mahalanobis_distance(X_test, X_train):
    cov_matrix = np.cov(X_train, rowvar=False) + np.eye(X_train.shape[1]) * 1e-4
    inv_cov_matrix = np.linalg.inv(cov_matrix)
            
    # Calcul via la fonction générique de sklearn avec l'inverse de la covariance (VI)
    dist_matrix = pairwise_distances(X_test, X_train, metric='mahalanobis', VI=inv_cov_matrix)

    return dist_matrix

DIST_DICT = {
        'euclidean': euclidean_distances,
        'manhattan': manhattan_distances,
        'cosine': cosine_distances,
        'mahalanobis': mahalanobis_distance
    }

class LOONearestCountryImpute:

    def __init__(
            self, 
            dataset: pd.DataFrame, 
            k_neighbours: int, 
            year: int, 
            distances: List[str] = ['euclidean'],
            chosen_features: List[str] = ['gdp_per_capita_constant_2015', 'rural_pop_percent', 'mean_temp', 'forest_area_hectares']):
        
        self.raw_dataset = dataset
        self.k_neighbours = k_neighbours
        self.year = year
        self.distances = distances
        self.chosen_features = chosen_features

        self.target = 'woodfuel_total_m3'
        self.mean_historical_prod = None
        self.country_list = None
        self.cross_sectional_df = None
        self.N_countries = None

        self.results = []

    
    def _preprocess_dataset(self):
        
        # Aggregating all woodfuel production
        self.raw_dataset['woodfuel_total_m3'] = (
            self.raw_dataset['woodfuel_coniferous_m3'].fillna(0) + 
            self.raw_dataset['woodfuel_non_coniferous_m3'].fillna(0)
        )

        # Computing know official (Label A) past production values
        official_past = self.raw_dataset[
            (self.raw_dataset['Year'] < self.year) & 
            (self.raw_dataset['Flag'] == 'A')
        ].dropna(subset=[self.target])

        self.mean_historical_prod = official_past.groupby('ISO3')[self.target].mean().to_dict()

        # Cross-sectional dataset on given year
        self.cross_sectional_df = self.raw_dataset[
            (self.raw_dataset['Year'] == self.year) & 
            (self.raw_dataset['Flag'] == 'A')
        ].dropna(subset=[self.target]).copy()

        self.country_list = self.cross_sectional_df['ISO3'].unique() 
        self.N_countries = len(self.country_list)


    def _loo_imputation(self, historical_scaling = True):

        for distance in self.distances: # Testing all given distances
            
            all_ground_truth = [] # List storing real/predicted values for each country left out
            all_preds = []
            
            # LOOCV
            for country in range(self.N_countries):

                test_country = self.country_list[country]
                
                df_test = self.cross_sectional_df[self.cross_sectional_df['ISO3'] == test_country].copy()
                df_train = self.cross_sectional_df[self.cross_sectional_df['ISO3'] != test_country].copy()
                
                real_prod_value = df_test[self.target].values[0]
                all_ground_truth.append(real_prod_value)
                
                # Re-scaling the data at each CV split (as the train/test change)
                scaler = StandardScaler()

                X_train = scaler.fit_transform(df_train[self.chosen_features])
                X_test = scaler.transform(df_test[self.chosen_features])

                                # Check if input features have NaNs
                if np.isnan(X_train).any():
                    print(f"--- WARNING: X_train contains NaNs on country index {country}! ---")
                    # Show which feature column has the NaN values
                    nan_cols = df_train[self.chosen_features].isna().sum()
                    print("NaN counts per feature column in df_train:")
                    print(nan_cols)
                
                # Computing the distance matrix
                dist_matrix = DIST_DICT[distance](X_test, X_train)[0]
                
                # TIsolating the k nearest neighbours
                chosen_neighbours = np.argsort(dist_matrix)[:self.k_neighbours]
                
                neighbour_preds = []
                neigbhours_weights = []
                
                # We compute the ratio 
                hist_prod_test = self.mean_historical_prod.get(test_country, np.nan)
                
                # For each neighbour, we find his production and scale it according to historical data
                for neigh in chosen_neighbours:
                    neigh_iso3 = df_train['ISO3'].iloc[neigh]
                    neigh_prod = df_train[self.target].iloc[neigh]
                    hist_prod_neigh = self.mean_historical_prod.get(neigh_iso3, np.nan)
                    
                    if historical_scaling == True and pd.notna(hist_prod_neigh) and pd.notna(hist_prod_test) and hist_prod_neigh > 0:
                        hist_ratio = hist_prod_test / hist_prod_neigh
                        neigh_pred = neigh_prod * hist_ratio
                    else:
                        neigh_pred = neigh_prod
                        
                    neighbour_preds.append(neigh_pred)
                    
                    # Weighting each neighbour contribution by his distance inverse
                    neigh_distance = dist_matrix[neigh]
                    neigh_weight = 1.0 / (neigh_distance + 1e-5)
                    neigbhours_weights.append(neigh_weight)
                
                # Final predicted production is weighted average
                prod_predite = np.average(neighbour_preds, weights=neigbhours_weights)
                all_preds.append(prod_predite)
                
            # Final prediction arrays for metrics
            y_true = np.array(all_ground_truth)
            y_pred = np.array(all_preds)
            
            real_values_mean = y_true.mean()
            mae = mean_absolute_error(y_true, y_pred)
            mae_relatif = (mae / real_values_mean) * 100
            r2 = r2_score(y_true, y_pred)
            
            mape_mask = y_true > 1000
            mape = np.mean(np.abs((y_true[mape_mask] - y_pred[mape_mask]) / y_true[mape_mask])) * 100
            
            # Sauvegarde des résultats
            self.results.append({
                'Distance': distance,
                'R2': round(r2, 3),
                'Absolute MAE': round(mae, 2),
                'Relative MAE (%)': round(mae_relatif, 2),
                'MAPE (%)': round(mape, 2)
            })
    
    def run_evaluation(self):

        self._preprocess_dataset()
        self._loo_imputation()
    

    def print_results(self):
        print(f"--- PARAMETERS USED ---")
        print(f"K Neigbours: {self.k_neighbours}")
        print(f"Features used: {self.chosen_features}")

        df_perf = pd.DataFrame(self.results)

        print("\n" + "="*65)
        print(f"  LOOCV IMPUTATION RESULTS({self.year})")
        print("="*65)
        print(df_perf.to_string(index=False))
        print("="*65)

    
    def plot_distance_comparison(self):

        df_perf = pd.DataFrame(self.results)

        fig, ax1 = plt.subplots(figsize=(11, 5))

        color = '#1f77b4'
        ax1.set_xlabel('Distances')
        ax1.set_ylabel('Global R²', color=color)
        bars = ax1.bar(df_perf['Distance'], df_perf['R2'], color=color, alpha=0.6, width=0.4, label='R² Global')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim(0, 1.0)

        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

        ax2 = ax1.twinx()  
        color = '#d62728'
        ax2.set_ylabel(' Relative MAE(%)', color=color)
        line = ax2.plot(df_perf['Distance'], df_perf['Relative MAE (%)'], color=color, marker='o', linewidth=2, label='Relative MAE (%)')
        ax2.tick_params(axis='y', labelcolor=color)

        for i, txt in enumerate(df_perf['Relative MAE (%)']):
            ax2.annotate(f'{txt:.1f}%', (df_perf['Distance'].iloc[i], df_perf['Relative MAE (%)'].iloc[i]),
                        textcoords="offset points", xytext=(0,10), ha='center', color=color, fontweight='bold')

        plt.title(f'Distances benchmark ({self.year})', fontsize=12, pad=15)
        fig.tight_layout()
        plt.show()