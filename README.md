## **USING KNN TO DETECT ANOMALIES IN POWER CONSUMPTION**

Application of the K-Nearest Neighbors (KNN) algorithm to detect anomalies in simulated power consumption data over a month-long period.

## **Introduction**

Anomaly detection, a critical task in various domains, aims to identify patterns that deviate from normal behavior. In industries such as finance, cybersecurity, and energy management, detecting anomalies can prevent fraud, ensure network security, or optimize resource usage. Specifically, in the context of power consumption, early detection of unusual patterns, such as sudden spikes or drops in energy usage, can help prevent equipment failures, optimize energy distribution, and reduce costs.

In this study, we explore the application of machine learning techniques for detecting anomalies in power consumption data. Anomaly detection poses unique challenges that necessitate a careful approach to enhance model performance and ensure accurate identification of outliers. Our research emphasizes the importance of a comprehensive evaluation process to assess the effectiveness of the applied techniques over a month, specifically through the detection of anomalies in each of the four weeks.

## **Results and Discussion**

In this section, we present the performance of the KNN anomaly detection model before and after optimization.

### A. Before Optimization

- Focus on default parameters: **K=5** (number of neighbors) and **Threshold=100** (distance threshold).
- Weekly anomaly detection results and plots provide insights into the model’s behavior and its strengths and weaknesses in detecting anomalies across different weeks of power usage data.

![image](https://github.com/user-attachments/assets/4034c5b5-d1a4-4f20-94a9-af057d112b26) 

### Anomaly Detection Across Weeks

The model was applied to detect anomalies in the simulated power usage data for each of the four weeks. The detected anomalies are reported along with the deviation from the expected power usage.

![Screenshot 2024-10-16 221722](https://github.com/user-attachments/assets/0514d6e0-e84b-48a1-a8ce-7aa3a21f63bc)

### B. After Optimization

After optimizing the KNN model's parameters to **K=7** and **Threshold=100**, significant improvements in performance were observed. Below are the updated evaluation metrics and a discussion of the model’s behavior across the four weeks of power usage data.

![image](https://github.com/user-attachments/assets/6f8c3738-90a4-41ed-8687-636545b2d78e) 

- **Accuracy improved to 96.96%**, reflecting better classification performance.
- **Recall rose to 0.5360**, meaning the model now detects over 53.6% of actual anomalies (up from 22.50% pre-optimization).
- **F1-score increased to 0.6217**, indicating better balance between precision and recall.
- **AUC-ROC score slightly improved to 0.6893**, showing moderate ability to differentiate between normal and anomalous points.

### Anomaly Detection Across Weeks

With the optimized parameters, the model detected anomalies in Week 1 and Week 3, while Week 2 and Week 4 showed no anomalies, which matches the visual trends seen in the power usage plots.

![Screenshot 2024-10-16 220839](https://github.com/user-attachments/assets/ade7c04e-8395-4450-9c45-7e6d1ffd1a49) 

### Power Consumption Over 4 Weeks (Before Optimization)

![Screenshot 2024-10-16 221331](https://github.com/user-attachments/assets/f15aea4b-7086-4265-8b23-4626ce7897a3)
![Screenshot 2024-10-16 221315](https://github.com/user-attachments/assets/d30f62b1-df1a-40ae-adb2-ede53c534890)
![Screenshot 2024-10-16 221253](https://github.com/user-attachments/assets/d0add4d8-7621-43df-a4b0-2e5f6d94a9b1)
![Screenshot 2024-10-16 221346](https://github.com/user-attachments/assets/f264ded7-6238-494e-a48d-f7f9d266c449)

### Power Consumption Over 4 Weeks (After Optimization)

![Screenshot 2024-10-16 220447](https://github.com/user-attachments/assets/c632a2cb-06e0-4b53-9fb1-00a1a966e118)
![Screenshot 2024-10-16 220432](https://github.com/user-attachments/assets/28dc665e-1b7e-480e-9a5c-9d108872abdc)
![Screenshot 2024-10-16 220406](https://github.com/user-attachments/assets/0008e462-83a8-435a-b29a-e2ae5493a55f)
![Screenshot 2024-10-16 220459](https://github.com/user-attachments/assets/43d78c67-2ec2-42fe-b129-c5cc549f6908)


## **Conclusion**

This study demonstrates the effectiveness of the KNN algorithm for anomaly detection in power consumption data, especially through optimization. 

- Before optimization, the model achieved **95.39% accuracy** with a **recall of 22.50%**, missing many true anomalies.
- After optimizing parameters, the model improved significantly, with **96.96% accuracy**, a **recall of 53.6%**, and an improved **F1-score**.

The visualizations, including power consumption plots over the weeks, clearly show the detected anomalies. These visual aids provide context for understanding detected anomalies against typical usage patterns, allowing stakeholders to make informed decisions based on real-time data analysis.


