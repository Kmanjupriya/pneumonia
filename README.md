# pneumonia
# 🫁 AI-Powered Clinical Pneumonia Diagnostic Portal

An enterprise-grade web platform for medical radiography analysis. This system uses a **Hybrid Deep Learning Model** to evaluate Chest X-rays for pneumonia detection, generates **Grad-CAM visual heatmaps** for clinical explainability, and enforces strict **input validation** and **role-based access controls (RBAC)**.

---

## 🌟 Key Features

* **Hybrid Deep Learning Pipeline:** Integrates a custom PyTorch architecture with CLAHE (Contrast Limited Adaptive Histogram Equalization) image enhancement for accurate diagnostic predictions.
* **Explainable AI (Grad-CAM):** Generates spatial heatmaps indicating exact lung opacity regions (Red = AI attention / tissue opacification, Blue = normal aeration).
* **Multi-Stage X-Ray Validation:** Uses a 3-layer heuristic filter (Color Saturation Delta, Dynamic Range Histogram, and Spatial Luminance Distribution) to reject non-radiograph uploads (e.g., color photos, selfies, flat images) before running inference.
* **SHA-256 Duplicate Detection:** Hashes uploaded scans to prevent duplicate medical records while offering overwrite options.
* **Role-Based Access Control (RBAC):**
  * **Administrator:** System oversight, doctor account management/blocking, and system-wide immutable audit logging.
  * **Doctor:** Diagnostic evaluation, patient report management, editing, and deletion.
  * **Patient (Public vs. Private):** View personal history and direct diagnostic self-evaluations (Public mode).
* **Comprehensive Audit Trail:** Logs all user actions, authentication attempts, report modifications, and deletions with Before & After state snapshots.
* **Clinical Assistant Chatbot:** Embedded API assistant for answering questions on Grad-CAM interpretation, role capabilities, and portal functionality.

---

## 🏗️ Tech Stack

* **Backend Framework:** Python 3.x, Flask
* **Deep Learning & Computer Vision:** PyTorch, Torchvision, OpenCV, PIL (Pillow), NumPy
* **Database:** SQLite3
* **Frontend:** HTML5, CSS3, JavaScript, Jinja2 Templates

---

## 📁 Project Structure

```text
.
├── app.py                      # Core Flask application, routes, and validation logic
├── model.py                    # PyTorch HybridPneumoniaModel, CLAHE, & Grad-CAM routines
├── pneumonia_hybrid_model.pth  # Trained PyTorch model weights file
├── clinical_portal.db          # SQLite database (auto-generated on initial launch)
├── static/
│   └── uploads/                # Saved original scans & generated heatmap overlays
└── templates/
    ├── login.html              # Authentication & registration interface
    ├── dashboard.html          # Dynamic portal dashboard by user role
    ├── report.html             # Detailed diagnostic report & Grad-CAM visualizer
    └── duplicate_warning.html  # Handling duplicate SHA-256 scan detection
```text

 

🚀 Getting Started1. PrerequisitesEnsure you have Python 3.8 or higher installed.2. Clone the RepositoryBashgit clone [https://github.com/your-username/pneumonia-diagnostic-portal.git](https://github.com/your-username/pneumonia-diagnostic-portal.git)
cd pneumonia-diagnostic-portal
3. Create a Virtual Environment & Install DependenciesBash# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install flask torch torchvision pillow numpy opencv-python
4. Model Weights SetupEnsure your trained model weights file (pneumonia_hybrid_model.pth) is placed in the root directory alongside app.py and model.py.5. Run the ApplicationBashpython app.py
Open your browser and navigate to: http://127.0.0.1:5000🔐 Default Access CredentialsUpon initialization, the database seeds default accounts for quick testing:RoleEmailPasswordAdministratoradmin@mail.com123Doctordr.smith@hospital.org123Doctordr.jones@hospital.org123PatientRegister via UI or login with any new emailSelf-created🛡️ Input Validation SystemThe portal automatically screens incoming files using validate_chest_xray() in app.py:Color Saturation Filter: Calculates RGB channel variances ($\Delta_{RG}, \Delta_{GB} > 10.0$) to reject full-color photographs.Contrast Standard Deviation: Rejects solid, blank, or text-heavy low-variance images ($\sigma < 20.0$).Anatomical Spatial Luminance Check: Ensures center lung fields display brighter relative luminance compared to outer border regions.📝 DisclaimerThis portal is designed as a clinical decision support tool and educational demonstration. It is not intended to replace professional diagnostic judgment by a qualified radiologist or medical practitioner.
