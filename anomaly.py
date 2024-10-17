import random
import numpy as np
import plotly.graph_objs as go
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import ParameterGrid

# 1. Simulate power usage and generate ground truth
"""
    Simulates weekly power usage data with potential anomalies.
    
    Parameters:
    - num_weeks: Number of weeks to simulate
    - noise_level: Level of noise to add to the power usage data
    - anomaly_prob: Probability of introducing an anomaly in usage
    
    Returns:
    - power_usage: Simulated power usage data
    - ground_truth: Ground truth labels indicating anomalies
"""
def simulate_weekly_power_usage(num_weeks=4, noise_level=50, anomaly_prob=0.05):
    power_usage = []
    ground_truth = []
    
    for week in range(num_weeks):
        weekly_pattern = []
        weekly_labels = []
        for day in range(7):
            daily_pattern = []
            daily_labels = []
            for hour in range(24):
                if 6 <= hour < 18:
                    base_usage = random.randint(300, 500)
                else:
                    base_usage = random.randint(100, 200)
                
                usage = base_usage + random.gauss(0, noise_level)
                label = 0
                
                if random.random() < anomaly_prob:
                    usage *= random.choice([0.5, 2.5])
                    label = 1
                
                daily_pattern.append(usage)
                daily_labels.append(label)
            weekly_pattern.append(daily_pattern)
            weekly_labels.append(daily_labels)
        power_usage.append(weekly_pattern)
        ground_truth.append(weekly_labels)
    
    return power_usage, ground_truth


# 2. KNN-based anomaly detection
"""
    Detects anomalies using K-Nearest Neighbors (KNN).
    
    Parameters:
    - data: Power usage data for a week
    - k: Number of neighbors to consider
    - threshold: Distance threshold for detecting anomalies
    
    Returns:
    - anomalies: Indices of detected anomalies
"""
def knn_anomaly_detection(data, k=5, threshold=100):
    data = np.array([hour for day in data for hour in day]).reshape(-1, 1)
    
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(data)
    
    distances, _ = nbrs.kneighbors(data)
    avg_distances = np.mean(distances, axis=1)
    
    anomalies = np.where(avg_distances > threshold)[0]
    
    return anomalies


# 3. Model evaluation using classification metrics
"""
    Evaluates the model performance using various classification metrics.
    
    Parameters:
    - ground_truth: True labels of the data
    - predictions: Predicted labels (anomalies detected)
    
    Returns:
    - A dictionary of evaluation metrics
"""
def evaluate_model(ground_truth, predictions):
    accuracy = accuracy_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions, zero_division=0)  # Handle undefined precision
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    auc = roc_auc_score(ground_truth, predictions)

    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'AUC-ROC': auc
    }


# 4. Parameter search for best k and threshold
"""
    Tunes KNN parameters (k and threshold) to find the best performing configuration.
    
    Parameters:
    - power_usage: Simulated power usage data
    - ground_truth: Ground truth labels indicating anomalies
    - param_grid: Dictionary of parameters to search over
    
    Returns:
    - best_params: Dictionary with the best parameters and their evaluation metrics
"""
def tune_knn_parameters(power_usage, ground_truth, param_grid):
    flattened_ground_truth = [label for week in ground_truth for day in week for label in day]  # Flatten ground truth
    best_score = 0
    best_params = None
    
    # Loop through each combination of parameters in the grid
    for params in ParameterGrid(param_grid):
        k = params['k']
        threshold = params['threshold']
        predictions = []
        
        # Run KNN for each week and get predictions
        for week in range(4):
            weekly_data = power_usage[week]
            anomalies = knn_anomaly_detection(weekly_data, k=k, threshold=threshold)
            
            # Generate a prediction array for the current week
            weekly_predictions = np.zeros(len(weekly_data) * 24)  # 24 hours for each day
            weekly_predictions[anomalies] = 1  # Mark anomalies as 1
            predictions.extend(weekly_predictions)
        
        # Evaluate model performance
        evaluation_results = evaluate_model(flattened_ground_truth, predictions)
        
        
        if evaluation_results['F1-Score'] > best_score:
            best_score = evaluation_results['F1-Score']
            best_params = {
                'k': k,
                'threshold': threshold,
                'evaluation': evaluation_results
            }
    
    return best_params


# 5. Visualization function with improved information display
"""
    Visualizes the weekly power usage data and highlights detected anomalies.
    
    Parameters:
    - power_usage: Simulated power usage data
    - anomalies_per_week: Detected anomalies for each week
"""
def visualize_weekly_data_improved(power_usage, anomalies_per_week):
    for week in range(4):
        weekly_data = [hour for day in power_usage[week] for hour in day]  # Flatten weekly data
        hourly_indices = list(range(len(weekly_data)))  # Create a list of hourly indices for the x-axis

        # Calculate weekly average and standard deviation
        average_usage = np.mean(weekly_data)
        std_dev = np.std(weekly_data)
        
        # Create the base line plot for power usage
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly_indices, y=weekly_data,
                                 mode='lines', name=f'Week {week + 1} Power Usage',
                                 line=dict(color='blue')))
        
        # Add average consumption line
        fig.add_trace(go.Scatter(x=hourly_indices, y=[average_usage]*len(hourly_indices),
                                 mode='lines', name=f'Average Consumption: {average_usage:.2f} kW',
                                 line=dict(color='orange', dash='dash')))
        
        # Add shaded areas to differentiate day (6 AM to 6 PM) and night periods
        for day in range(7):
            day_start = day * 24 + 6  # 6 AM
            day_end = day * 24 + 18   # 6 PM
            fig.add_vrect(x0=day_start, x1=day_end, fillcolor="lightgreen", opacity=0.2, layer="below", 
                          line_width=0, annotation_text="Daytime", annotation_position="top left")

        # If there are anomalies for this week, plot them with hover info and print details
        if week in anomalies_per_week and len(anomalies_per_week[week]) > 0:
            anomalies = anomalies_per_week[week]
            anomaly_values = [weekly_data[i] for i in anomalies]  # Get the actual power values of the anomalies
            
            # Add anomaly points to the plot with detailed hover information
            fig.add_trace(go.Scatter(
                x=anomalies, y=anomaly_values,
                mode='markers', name='Anomalies',
                marker=dict(color='red', size=10, symbol='x'),
                hovertext=[f"Anomaly detected<br>Power Usage: {val:.2f} kW<br>Average: {average_usage:.2f} kW" 
                           for val in anomaly_values],  # Add hover info
                hoverinfo="text"
            ))
            
            # Print anomaly information (hour and power usage)
            print(f"\nAnomalies in Week {week + 1}:")
            for i, val in zip(anomalies, anomaly_values):
                print(f"Hour: {i}, Power Usage: {val:.2f} kW, Deviation: {abs(val - average_usage):.2f} kW")

        # Set titles and labels
        fig.update_layout(title=f"Week {week + 1}: Power Usage with Anomaly Detection",
                          xaxis_title="Hour", yaxis_title="Power Usage (kW)",
                          hovermode='x unified')

        # Add a summary annotation for the weekly average and standard deviation
        fig.add_annotation(
            text=f"Weekly Average: {average_usage:.2f} kW<br>Std Dev: {std_dev:.2f} kW",
            xref="paper", yref="paper", x=0.9, y=1.1, showarrow=False,
            bordercolor="black", borderwidth=1
        )
        
        # Show the plot
        fig.show()


# 6. Main execution

# Simulate power usage for 4 weeks (1 month)
power_usage, ground_truth = simulate_weekly_power_usage(num_weeks=4)

# Define the parameter grid for k and threshold
param_grid = {
    'k': [3, 5, 7, 9, 12],  # Range of k values (number of neighbors)
    'threshold': [50, 75, 100, 150, 200]  # Range of thresholds
}

# Tune KNN parameters
best_params = tune_knn_parameters(power_usage, ground_truth, param_grid)

# Check if best_params has valid values before accessing them
if best_params is not None:
    print("\nBest Parameters:")
    print(f"K (Neighbors): {best_params['k']}")
    print(f"Threshold: {best_params['threshold']}")
    print("\nEvaluation Metrics with Best Parameters:")
    for metric, value in best_params['evaluation'].items():
        print(f'{metric}: {value:.4f}')
    
    # Visualize weekly data and highlight anomalies
    anomalies_per_week = {}

    for week in range(4):
        weekly_data = power_usage[week]
        anomalies = knn_anomaly_detection(weekly_data, k=best_params['k'], threshold=best_params['threshold'])
        
        # Store detected anomalies for visualization
        if len(anomalies) > 0:
            anomalies_per_week[week] = anomalies

    # Visualize the results with improved visualization
    visualize_weekly_data_improved(power_usage, anomalies_per_week)
else:
    print("No valid parameters found during grid search.")
