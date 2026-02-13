import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("pcos_50000_dataset.csv")

X = df.drop("PCOS", axis=1)
y = df["PCOS"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# GUI Window
root = tk.Tk()
root.title("PCOS Prediction System")
root.geometry("400x550")

tk.Label(root, text="PCOS Prediction System", font=("Arial", 16, "bold")).pack(pady=10)

fields = [
    "Age", "BMI", "Hairfall", "Acne",
    "Irregular_Cycle", "Weight_Gain",
    "Hirsutism", "FSH", "LH", "AMH",
    "Cycle_Length"
]

entries = {}

for field in fields:
    tk.Label(root, text=field).pack()
    entry = tk.Entry(root)
    entry.pack()
    entries[field] = entry

def predict():
    try:
        values = [float(entries[field].get()) for field in fields]
        input_data = np.array(values).reshape(1, -1)
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            result = "⚠ PCOS Detected"
        else:
            result = "✓ No PCOS"

        messagebox.showinfo("Result", result)

    except:
        messagebox.showerror("Error", "Please enter valid numbers!")

tk.Button(root, text="Predict", command=predict, bg="green", fg="white").pack(pady=20)

root.mainloop()
