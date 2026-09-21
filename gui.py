import tkinter as tk
from tkinter import ttk, messagebox
import joblib
import numpy as np
from tensorflow.keras.models import load_model


# ============================================================
# LOAD TRAINED MODEL AND SCALER
# ============================================================

model = load_model("heart_model.keras")
scaler = joblib.load("scaler.pkl")


# ============================================================
# MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("Heart Disease Prediction System")
root.geometry("700x900")
root.configure(bg="#f5f7fa")
root.resizable(False, False)


# ============================================================
# TITLE
# ============================================================

title = tk.Label(
    root,
    text="❤️ Heart Disease Prediction System",
    font=("Arial", 22, "bold"),
    fg="#c0392b",
    bg="#f5f7fa"
)

title.pack(pady=(20, 5))


subtitle = tk.Label(
    root,
    text="AI-Based Medical Prediction Prototype",
    font=("Arial", 11),
    fg="gray",
    bg="#f5f7fa"
)

subtitle.pack()


# ============================================================
# MAIN FRAME
# ============================================================

frame = tk.Frame(
    root,
    bg="white",
    padx=25,
    pady=20,
    relief="groove",
    bd=2
)

frame.pack(
    padx=20,
    pady=15,
    fill="both",
    expand=True
)


# ============================================================
# VARIABLES
# ============================================================

age = tk.StringVar()
gender = tk.StringVar()
cp = tk.StringVar()
trestbps = tk.StringVar()
chol = tk.StringVar()
fbs = tk.StringVar()
restecg = tk.StringVar()
thalach = tk.StringVar()
exang = tk.StringVar()
oldpeak = tk.StringVar()
slope = tk.StringVar()
ca = tk.StringVar()
thal = tk.StringVar()


# ============================================================
# FONT
# ============================================================

font_label = ("Arial", 11)
font_entry = ("Arial", 11)


# ============================================================
# AGE
# ============================================================

tk.Label(
    frame,
    text="Age",
    bg="white",
    font=font_label
).grid(row=0, column=0, sticky="w", pady=6)

tk.Entry(
    frame,
    textvariable=age,
    font=font_entry,
    width=25
).grid(row=0, column=1)


# ============================================================
# GENDER
# ============================================================

tk.Label(
    frame,
    text="Gender",
    bg="white",
    font=font_label
).grid(row=1, column=0, sticky="w", pady=6)

ttk.Combobox(
    frame,
    textvariable=gender,
    values=["Male", "Female"],
    state="readonly",
    width=22
).grid(row=1, column=1)


# ============================================================
# CHEST PAIN TYPE
# ============================================================

tk.Label(
    frame,
    text="Chest Pain Type",
    bg="white",
    font=font_label
).grid(row=2, column=0, sticky="w", pady=6)

ttk.Combobox(
    frame,
    textvariable=cp,
    values=[
        "Typical Angina",
        "Atypical Angina",
        "Non-anginal Pain",
        "Asymptomatic"
    ],
    state="readonly",
    width=22
).grid(row=2, column=1)


# ============================================================
# BLOOD PRESSURE
# ============================================================

tk.Label(
    frame,
    text="Blood Pressure",
    bg="white",
    font=font_label
).grid(row=3, column=0, sticky="w", pady=6)

tk.Entry(
    frame,
    textvariable=trestbps,
    width=25
).grid(row=3, column=1)


# ============================================================
# CHOLESTEROL
# ============================================================

tk.Label(
    frame,
    text="Cholesterol",
    bg="white",
    font=font_label
).grid(row=4, column=0, sticky="w", pady=6)

tk.Entry(
    frame,
    textvariable=chol,
    width=25
).grid(row=4, column=1)


# ============================================================
# FASTING BLOOD SUGAR
# ============================================================

tk.Label(
    frame,
    text="Fasting Blood Sugar",
    bg="white",
    font=font_label
).grid(row=5, column=0, sticky="w", pady=6)

ttk.Combobox(
    frame,
    textvariable=fbs,
    values=["No", "Yes"],
    state="readonly",
    width=22
).grid(row=5, column=1)


# ============================================================
# ECG RESULT
# ============================================================

tk.Label(
    frame,
    text="ECG Result",
    bg="white",
    font=font_label
).grid(row=6, column=0, sticky="w", pady=6)

ttk.Combobox(
    frame,
    textvariable=restecg,
    values=[
        "Normal",
        "ST-T Abnormality",
        "Left Ventricular Hypertrophy"
    ],
    state="readonly",
    width=22
).grid(row=6, column=1)


# ============================================================
# MAXIMUM HEART RATE
# ============================================================

tk.Label(
    frame,
    text="Maximum Heart Rate",
    bg="white",
    font=font_label
).grid(row=7, column=0, sticky="w", pady=6)

tk.Entry(
    frame,
    textvariable=thalach,
    width=25
).grid(row=7, column=1)


# ============================================================
# EXERCISE ANGINA
# ============================================================

tk.Label(
    frame,
    text="Exercise Angina",
    bg="white",
    font=font_label
).grid(row=8, column=0, sticky="w", pady=6)

ttk.Combobox(
    frame,
    textvariable=exang,
    values=["No", "Yes"],
    state="readonly",
    width=22
).grid(row=8, column=1)


# ============================================================
# OLD PEAK
# ============================================================

tk.Label(
    frame,
    text="Old Peak",
    bg="white",
    font=font_label
).grid(row=9, column=0, sticky="w", pady=6)

tk.Entry(
    frame,
    textvariable=oldpeak,
    width=25
).grid(row=9, column=1)


# ============================================================
# SLOPE
# ============================================================

tk.Label(
    frame,
    text="Slope",
    bg="white",
    font=font_label
).grid(row=10, column=0, sticky="w", pady=6)

ttk.Combobox(
    frame,
    textvariable=slope,
    values=[
        "Upsloping",
        "Flat",
        "Downsloping"
    ],
    state="readonly",
    width=22
).grid(row=10, column=1)


# ============================================================
# MAJOR VESSELS
# ============================================================

tk.Label(
    frame,
    text="Major Vessels",
    bg="white",
    font=font_label
).grid(row=11, column=0, sticky="w", pady=6)

ttk.Combobox(
    frame,
    textvariable=ca,
    values=["0", "1", "2", "3", "4"],
    state="readonly",
    width=22
).grid(row=11, column=1)


# ============================================================
# THALASSEMIA
# ============================================================

tk.Label(
    frame,
    text="Thalassemia",
    bg="white",
    font=font_label
).grid(row=12, column=0, sticky="w", pady=6)

ttk.Combobox(
    frame,
    textvariable=thal,
    values=[
        "Normal",
        "Fixed Defect",
        "Reversible Defect"
    ],
    state="readonly",
    width=22
).grid(row=12, column=1)


# ============================================================
# RESULT LABEL
# ============================================================

result_label = tk.Label(
    frame,
    text="Prediction Result",
    font=("Arial", 14, "bold"),
    bg="white",
    fg="blue",
    width=30,
    height=2,
    justify="center"
)

result_label.grid(
    row=15,
    column=0,
    columnspan=2,
    pady=15
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict():

    try:

        # ----------------------------------------------------
        # Convert Gender
        # ----------------------------------------------------

        sex = 1 if gender.get() == "Male" else 0


        # ----------------------------------------------------
        # Chest Pain Mapping
        # ----------------------------------------------------

        cp_map = {
            "Typical Angina": 0,
            "Atypical Angina": 1,
            "Non-anginal Pain": 2,
            "Asymptomatic": 3
        }


        # ----------------------------------------------------
        # Fasting Blood Sugar
        # ----------------------------------------------------

        fbs_map = {
            "No": 0,
            "Yes": 1
        }


        # ----------------------------------------------------
        # ECG Mapping
        # ----------------------------------------------------

        restecg_map = {
            "Normal": 0,
            "ST-T Abnormality": 1,
            "Left Ventricular Hypertrophy": 2
        }


        # ----------------------------------------------------
        # Exercise Angina
        # ----------------------------------------------------

        exang_map = {
            "No": 0,
            "Yes": 1
        }


        # ----------------------------------------------------
        # Slope Mapping
        # ----------------------------------------------------

        slope_map = {
            "Upsloping": 0,
            "Flat": 1,
            "Downsloping": 2
        }


        # ----------------------------------------------------
        # Thalassemia Mapping
        # ----------------------------------------------------

        thal_map = {
            "Normal": 1,
            "Fixed Defect": 2,
            "Reversible Defect": 3
        }


        # ----------------------------------------------------
        # Create Patient Data
        # EXACT ORDER OF 13 MODEL FEATURES
        # ----------------------------------------------------

        patient = [[

            float(age.get()),

            sex,

            cp_map[cp.get()],

            float(trestbps.get()),

            float(chol.get()),

            fbs_map[fbs.get()],

            restecg_map[restecg.get()],

            float(thalach.get()),

            exang_map[exang.get()],

            float(oldpeak.get()),

            slope_map[slope.get()],

            int(ca.get()),

            thal_map[thal.get()]

        ]]


        # ----------------------------------------------------
        # Convert to NumPy Array
        # ----------------------------------------------------

        patient = np.array(patient)


        # ----------------------------------------------------
        # Apply Same Scaling Used During Training
        # ----------------------------------------------------

        patient_scaled = scaler.transform(patient)


        # ----------------------------------------------------
        # Model Prediction
        # ----------------------------------------------------

        prediction = model.predict(
            patient_scaled,
            verbose=0
        )


        probability = prediction[0][0]


        # ----------------------------------------------------
        # Display ONLY Prediction
        # ----------------------------------------------------

        if probability >= 0.5:

            result_label.config(
                text="🔴 Heart Disease Detected",
                fg="red"
            )

        else:

            result_label.config(
                text="🟢 No Heart Disease Detected",
                fg="green"
            )


    except ValueError:

        messagebox.showerror(
            "Invalid Input",
            "Please enter valid numbers in all numeric fields."
        )


    except KeyError:

        messagebox.showerror(
            "Missing Selection",
            "Please select an option from every dropdown."
        )


    except Exception as e:

        messagebox.showerror(
            "Error",
            f"Something went wrong:\n\n{e}"
        )


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_btn = tk.Button(
    frame,
    text="🔍 Predict",
    font=("Arial", 13, "bold"),
    bg="#27ae60",
    fg="white",
    width=18,
    height=1,
    command=predict
)

predict_btn.grid(
    row=14,
    column=0,
    columnspan=2,
    pady=10
)


# ============================================================
# RESET FUNCTION
# ============================================================

def clear_fields():

    age.set("")
    gender.set("")
    cp.set("")
    trestbps.set("")
    chol.set("")
    fbs.set("")
    restecg.set("")
    thalach.set("")
    exang.set("")
    oldpeak.set("")
    slope.set("")
    ca.set("")
    thal.set("")

    result_label.config(
        text="Prediction Result",
        fg="blue"
    )


# ============================================================
# RESET BUTTON
# ============================================================

reset_btn = tk.Button(
    frame,
    text="🔄 Reset",
    font=("Arial", 13, "bold"),
    bg="#e67e22",
    fg="white",
    width=18,
    height=1,
    command=clear_fields
)

reset_btn.grid(
    row=16,
    column=0,
    columnspan=2,
    pady=5
)


# ============================================================
# START GUI
# ============================================================

root.mainloop()