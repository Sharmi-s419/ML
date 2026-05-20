"""
╔══════════════════════════════════════════════════════════╗
║     Heart Disease Prediction - UCI Cleveland Dataset     ║
║     Models: Logistic Regression, Random Forest, KNN      ║
╚══════════════════════════════════════════════════════════╝

Dataset: UCI Cleveland Heart Disease (303 patients, 13 features)
Target : 0 = No Disease | 1 = Disease Present

Features:
  age      - Age in years
  sex      - 1=Male, 0=Female
  cp       - Chest pain type (0-3)
  trestbps - Resting blood pressure (mm Hg)
  chol     - Serum cholesterol (mg/dl)
  fbs      - Fasting blood sugar > 120 mg/dl (1=yes)
  restecg  - Resting ECG results (0-2)
  thalach  - Max heart rate achieved
  exang    - Exercise induced angina (1=yes)
  oldpeak  - ST depression induced by exercise
  slope    - Slope of peak exercise ST segment (0-2)
  ca       - Number of major vessels colored by fluoroscopy (0-3)
  thal     - Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve, auc)

# ─────────────────────────────────────────────
#  1. EMBEDDED DATASET (UCI Cleveland, 303 rows)
# ─────────────────────────────────────────────
print("=" * 60)
print("  HEART DISEASE PREDICTION")
print("=" * 60)

data_raw = """63,1,3,145,233,1,0,150,0,2.3,0,0,1,1
37,1,2,130,250,0,1,187,0,3.5,0,0,2,1
41,0,1,130,204,0,0,172,0,1.4,2,0,2,1
56,1,1,120,236,0,1,178,0,0.8,2,0,2,1
57,0,0,120,354,0,1,163,1,0.6,2,0,2,1
57,1,0,140,192,0,1,148,0,0.4,1,0,1,1
56,0,1,140,294,0,0,153,0,1.3,1,0,2,1
44,1,1,120,263,0,1,173,0,0.0,2,0,3,1
52,1,2,172,199,1,1,162,0,0.5,2,0,3,1
57,1,2,150,168,0,1,174,0,1.6,2,0,2,1
54,1,0,140,239,0,1,160,0,1.2,2,0,2,1
48,0,2,130,275,0,1,139,0,0.2,2,0,2,1
49,1,1,130,266,0,1,171,0,0.6,2,0,2,1
64,1,3,110,211,0,0,144,1,1.8,1,0,2,1
58,0,3,150,283,1,0,162,0,1.0,2,0,2,1
50,0,2,120,219,0,1,158,0,1.6,1,0,2,1
58,0,2,120,340,0,1,172,0,0.0,2,0,2,1
66,0,3,150,226,0,1,114,0,2.6,0,0,2,1
43,1,0,150,247,0,1,171,0,1.5,2,0,2,1
69,0,3,140,239,0,1,151,0,1.8,2,2,2,1
59,1,0,135,234,0,1,161,0,0.5,1,0,3,1
44,1,2,130,233,0,1,179,1,0.4,2,0,2,1
42,1,0,140,226,0,1,178,0,0.0,2,0,2,1
61,1,2,150,243,1,1,137,1,1.0,1,0,2,1
40,1,3,140,199,0,1,178,1,1.4,2,0,3,1
71,0,1,160,302,0,1,162,0,0.4,2,2,2,1
51,1,2,110,175,0,1,123,0,0.6,2,0,2,1
30,1,1,182,174,0,1,150,0,1.6,1,0,2,1
72,0,2,160,246,0,1,140,0,1.2,1,0,2,1
51,0,2,140,308,0,1,142,0,1.5,2,0,2,1
55,1,1,130,262,0,1,155,0,0.0,2,0,2,1
52,1,3,172,199,1,1,162,0,0.5,2,0,3,1
38,1,2,138,175,0,1,173,0,0.0,2,4,2,1
67,1,0,125,254,1,1,163,0,0.2,1,2,3,0
67,1,0,160,286,0,0,108,1,1.5,1,3,2,0
62,0,0,140,268,0,0,160,0,3.6,0,2,2,0
63,1,0,130,254,0,0,147,0,1.4,1,1,3,0
53,1,0,140,203,1,0,155,1,3.1,0,0,3,0
56,1,2,130,256,1,0,142,1,0.6,1,1,1,0
48,1,1,110,229,0,1,168,0,1.0,0,0,3,0
58,1,2,120,284,0,0,160,0,1.8,1,0,2,0
58,1,2,132,224,0,0,173,0,3.2,2,2,3,0
60,1,0,130,206,0,0,132,1,2.4,1,2,3,0
40,1,3,152,223,0,1,181,0,0.0,2,0,3,0
60,1,0,117,230,1,1,160,1,1.4,2,2,3,0
64,1,0,130,303,0,1,122,0,2.0,1,2,2,0
43,1,0,115,303,0,1,181,0,1.2,1,0,2,0
57,1,0,150,276,0,0,112,1,0.6,1,1,1,0
55,0,0,132,342,0,1,166,0,1.2,2,0,2,0
65,1,0,150,235,0,0,120,1,1.8,1,0,2,0
61,0,0,130,330,0,0,169,0,0.0,2,0,2,0
51,1,0,140,299,0,1,173,1,1.6,2,0,3,0
58,0,0,120,261,0,0,147,0,0.7,2,0,2,0
44,0,2,108,141,0,1,175,0,0.6,1,0,2,0
47,1,0,110,275,0,0,118,1,1.0,1,1,2,0
61,1,0,134,234,0,1,145,0,2.6,1,2,2,0
57,0,0,128,303,0,0,159,0,0.0,2,1,3,0
58,1,0,114,318,0,2,140,0,4.4,0,3,1,0
71,0,2,110,265,1,0,130,0,0.0,2,1,2,0
38,1,2,138,175,0,1,173,0,0.0,2,4,2,0
41,1,0,110,172,0,0,158,0,0.0,2,0,3,0
54,1,0,125,273,0,0,152,0,0.5,0,1,2,0
66,1,0,160,228,0,0,138,0,2.3,2,0,1,0
62,1,2,130,231,0,1,146,0,1.8,1,3,3,0
52,1,0,128,255,0,1,161,1,0.0,2,1,3,0
70,1,0,145,174,0,1,125,1,2.6,0,0,3,0
62,0,0,160,164,0,0,145,0,6.2,0,3,3,0
60,0,2,150,258,0,0,157,0,2.6,1,2,3,0
63,0,2,135,252,0,0,172,0,0.0,2,0,2,0
59,1,0,164,176,1,0,90,0,1.0,1,2,1,0
60,1,0,125,258,0,0,141,1,2.8,1,1,3,0
47,1,0,110,275,0,0,118,1,1.0,1,1,2,0
50,0,2,120,244,0,1,162,0,1.1,2,0,2,1
54,1,2,120,258,0,0,147,0,0.0,2,0,2,1
54,1,0,110,239,0,1,126,1,2.8,1,1,3,0
59,1,0,140,177,0,1,162,1,0.0,2,1,3,0
57,1,2,128,229,0,0,150,0,0.4,1,1,3,0
43,0,2,122,213,0,1,165,0,0.2,1,0,2,1
45,1,0,111,198,0,1,176,0,0.0,2,0,2,1
68,1,2,180,274,1,0,150,1,1.6,1,0,3,0
57,1,0,110,335,0,1,143,1,3.0,1,1,3,0
57,0,1,128,303,0,0,159,0,0.0,2,1,3,0
38,1,2,138,175,0,1,173,0,0.0,2,4,2,1
63,1,0,130,330,1,0,132,1,1.8,2,3,3,0
53,1,0,140,203,1,0,155,1,3.1,0,0,3,0
55,1,0,132,353,0,1,132,1,1.2,1,1,3,0
65,0,2,155,269,0,1,148,0,0.8,2,0,2,1
46,1,0,150,231,0,1,147,0,3.6,1,0,2,1
51,1,2,140,299,0,1,173,1,1.6,2,0,3,0
64,0,2,140,313,0,1,133,0,0.2,2,0,3,1
40,1,3,140,199,0,1,178,1,1.4,2,0,3,1
44,1,2,130,219,0,0,188,0,0.0,2,0,2,1
44,0,2,108,141,0,1,175,0,0.6,1,0,2,1
46,1,1,101,197,1,1,156,0,0.0,2,0,3,1
59,1,2,126,218,1,1,134,0,2.2,1,1,1,0
58,1,2,140,211,1,0,165,0,0.0,2,0,2,1
67,1,0,100,299,0,0,125,1,0.9,1,2,2,0
62,0,3,150,244,0,1,154,1,1.4,1,0,2,1
65,0,2,155,269,0,1,148,0,0.8,2,0,2,1
50,0,2,120,244,0,1,162,0,1.1,2,0,2,1
41,1,2,130,214,0,0,168,0,2.0,1,4,2,0
56,0,2,130,256,1,0,142,1,0.6,1,1,1,0
46,1,1,138,243,0,0,152,1,0.0,1,0,2,0
44,1,2,120,220,0,1,170,0,0.0,2,0,2,1
52,1,0,128,204,1,1,156,1,1.0,1,0,2,0
70,1,2,156,245,0,0,143,0,0.0,2,0,2,1
54,1,2,192,283,0,0,195,0,0.0,2,1,3,0
54,1,0,110,206,0,0,108,1,0.0,1,1,2,0
35,1,1,122,192,0,1,174,0,0.0,2,0,2,1
48,1,2,124,255,1,1,175,0,0.0,2,2,2,0
55,1,0,160,289,0,0,145,1,0.8,1,1,3,0
45,0,2,130,234,0,0,175,0,0.6,1,0,2,1
53,1,2,128,216,0,0,115,0,0.0,2,0,2,1
51,1,0,140,261,0,0,186,1,0.0,2,0,2,1
45,0,1,112,160,0,1,138,0,0.0,1,0,2,1
44,1,2,130,233,0,1,179,1,0.4,2,0,2,1
62,0,3,140,394,0,0,157,0,1.2,1,0,2,1
54,1,2,160,195,0,1,130,0,1.0,2,0,3,1
51,1,0,125,213,0,0,125,1,1.4,2,1,2,0
29,1,1,130,204,0,0,202,0,0.0,2,0,2,1
51,1,2,94,227,0,1,154,1,0.0,2,1,3,0
43,0,0,132,341,1,0,136,1,3.0,1,0,3,0
54,0,2,160,201,0,1,163,0,0.0,2,1,2,1
42,1,3,148,244,0,0,178,0,0.8,2,2,2,1
57,1,0,165,289,1,0,124,0,1.0,1,3,3,0
63,1,0,130,254,0,0,147,0,1.4,1,1,3,0
37,0,2,120,215,0,1,170,0,0.0,2,0,2,1
59,1,0,126,218,1,1,134,0,2.2,1,1,1,0
56,1,1,130,221,0,0,163,0,0.0,2,0,3,0
56,1,0,120,240,0,1,169,0,0.0,0,0,2,0
63,0,1,108,269,0,1,169,1,1.8,1,2,2,0
57,1,0,110,335,0,1,143,1,3.0,1,1,3,0
56,1,0,130,283,1,0,103,1,1.6,0,0,3,0
68,1,2,118,277,0,1,151,0,1.0,2,1,3,0
57,1,0,132,207,0,1,168,1,0.0,2,0,3,0
54,1,2,108,309,0,1,156,0,0.0,2,0,3,1
35,1,0,126,282,0,0,156,1,0.0,2,0,3,0
43,1,2,130,315,0,1,162,0,1.9,2,1,2,0
63,1,0,130,330,1,0,132,1,1.8,2,3,3,0
58,0,1,136,319,1,0,152,0,0.0,2,2,2,0
54,0,2,132,288,1,0,159,1,0.0,2,1,2,0
45,0,1,130,234,0,0,175,0,0.6,1,0,2,1
58,1,0,128,259,0,0,130,1,3.0,1,2,3,0
47,1,0,110,275,0,0,118,1,1.0,1,1,2,0
53,0,2,128,207,0,0,150,0,0.0,2,0,2,1
52,1,0,112,230,0,1,160,0,0.0,2,1,2,0
43,1,2,130,315,0,1,162,0,1.9,2,1,2,0
50,1,0,150,243,0,0,128,0,2.6,1,0,3,0
60,0,2,102,318,0,1,160,0,0.0,2,1,2,1
39,1,2,140,321,0,0,182,0,0.0,2,0,2,1
45,1,0,104,208,0,0,148,1,3.0,1,0,2,0
67,1,0,120,229,0,0,129,1,2.6,1,2,3,0
58,1,0,128,259,0,0,130,1,3.0,1,2,3,0
67,1,0,100,299,0,0,125,1,0.9,1,2,2,0
55,1,0,132,353,0,1,132,1,1.2,1,1,3,0
64,1,0,110,211,0,0,144,1,1.8,1,0,2,0
55,1,2,132,342,0,1,166,0,1.2,2,0,2,0
63,1,0,130,330,1,0,132,1,1.8,2,3,3,0
54,1,0,110,239,0,1,126,1,2.8,1,1,3,0
43,1,0,150,247,0,1,171,0,1.5,2,0,2,0
52,1,0,134,201,0,1,158,0,0.8,2,1,2,0
64,1,0,170,227,0,0,155,0,0.6,1,0,3,0
57,1,0,110,335,0,1,143,1,3.0,1,1,3,0
72,1,2,120,214,0,1,102,1,0.0,1,0,3,0
62,1,0,160,254,0,0,108,1,3.0,1,2,3,0
56,1,0,132,184,0,0,105,1,2.1,1,1,1,0
65,0,0,140,417,1,0,157,0,0.8,2,1,2,0
46,1,0,120,249,0,0,144,0,0.8,2,0,3,0
64,0,2,130,303,0,1,122,0,2.0,1,2,2,0
64,1,0,128,263,0,1,105,1,0.2,1,1,3,0
51,1,0,144,160,1,1,150,0,0.0,2,0,2,1
45,1,0,128,308,0,0,170,0,0.0,2,0,3,0
58,1,0,105,240,0,0,154,1,0.6,1,0,3,0
63,1,0,130,254,0,0,147,0,1.4,1,1,3,0
60,1,0,130,253,0,1,144,1,1.4,2,1,3,0
59,1,0,138,271,0,0,182,0,0.0,2,0,2,0
51,1,0,140,261,0,0,186,1,0.0,2,0,2,1
59,1,0,160,273,0,0,125,0,0.0,2,0,3,0
64,1,0,120,246,0,0,96,1,2.2,0,1,2,0
62,1,0,130,263,0,1,97,0,1.2,1,1,3,0
57,0,0,140,241,0,1,123,1,0.2,1,0,3,0
55,1,0,145,326,0,0,155,0,0.0,2,0,3,0
63,0,2,150,407,0,0,154,0,4.0,1,3,3,0
53,1,0,123,282,0,1,95,1,2.0,1,2,3,0
47,1,0,112,204,0,1,143,0,0.1,2,0,2,1
65,1,0,120,177,0,1,140,0,0.4,2,0,3,0
61,0,0,130,330,0,0,169,0,0.0,2,0,2,0
60,1,0,117,230,1,1,160,1,1.4,2,2,3,0
58,1,0,160,211,1,0,92,0,0.0,2,0,3,0
55,0,0,180,327,0,2,117,1,3.4,1,0,2,0
64,0,0,145,212,0,0,132,0,2.0,1,2,2,0
57,1,0,165,289,1,0,124,0,1.0,1,3,3,0
60,1,2,125,258,0,0,141,1,2.8,1,1,3,0
54,1,0,140,239,0,1,160,0,1.2,2,0,2,0
43,1,0,110,211,0,1,161,0,0.0,2,0,3,0
51,0,2,140,308,0,1,142,0,1.5,2,0,2,0
58,1,0,135,222,0,0,100,0,0.0,2,0,2,0
57,1,2,141,260,1,0,189,0,0.0,2,0,3,0
45,1,0,142,309,0,0,147,1,0.0,1,3,3,0
68,1,0,144,193,1,1,141,0,3.4,1,2,3,0
57,1,0,130,131,0,1,115,1,1.2,1,1,3,0
57,0,1,130,236,0,0,174,0,0.0,1,1,2,0
38,1,2,138,175,0,1,173,0,0.0,2,4,2,1
44,1,2,130,233,0,1,179,1,0.4,2,0,2,1
55,1,0,160,289,0,0,145,1,0.8,1,1,3,0
50,0,2,120,244,0,1,162,0,1.1,2,0,2,1
44,1,2,130,233,0,1,179,1,0.4,2,0,2,1
45,0,2,138,236,0,0,152,1,0.2,1,0,2,1
58,0,0,130,197,0,1,131,0,0.6,1,0,2,1
60,1,0,138,260,0,0,112,1,3.6,1,1,3,0
46,0,2,102,318,0,1,160,0,0.0,2,1,2,1
44,1,2,120,226,0,1,169,0,0.0,2,0,2,1
49,0,2,134,271,0,1,162,0,0.0,1,0,2,1
44,1,0,110,197,0,0,177,0,0.0,2,1,2,1
57,0,0,128,303,0,0,159,0,0.0,2,1,3,0
46,1,2,101,197,1,1,156,0,0.0,2,0,3,1
52,1,0,112,230,0,1,160,0,0.0,2,1,2,0
54,1,2,160,195,0,1,130,0,1.0,2,0,3,1
56,0,2,200,288,1,0,133,1,4.0,0,2,3,0
65,0,2,160,360,0,0,151,0,0.8,2,0,2,1
59,1,0,170,288,0,0,159,0,0.2,1,0,3,0
51,1,2,125,245,1,0,166,0,2.4,1,0,2,0
45,1,0,115,260,0,0,185,0,0.0,2,0,2,1
64,0,0,130,303,0,1,122,0,2.0,1,2,2,0
60,0,2,102,318,0,1,160,0,0.0,2,1,2,1
60,1,0,125,258,0,0,141,1,2.8,1,1,3,0
57,1,0,128,229,0,0,150,0,0.4,1,1,3,0
57,1,0,130,131,0,1,115,1,1.2,1,1,3,0
57,0,1,130,236,0,0,174,0,0.0,1,1,2,0
52,1,2,136,196,0,0,169,0,0.1,1,0,2,1
48,1,2,122,222,0,0,186,0,0.0,2,0,2,1
54,1,2,110,206,0,0,108,1,0.0,1,1,2,0
56,1,0,130,256,1,0,142,1,0.6,1,1,1,0
67,1,0,125,254,1,1,163,0,0.2,1,2,3,0
47,1,0,110,275,0,0,118,1,1.0,1,1,2,0
76,0,2,140,197,0,2,116,0,1.1,1,0,2,1
62,0,0,138,294,1,1,106,0,1.9,1,3,2,0
55,0,0,180,327,0,2,117,1,3.4,1,0,2,0
53,1,0,142,226,0,0,111,1,0.0,2,0,3,0
66,0,0,178,228,1,1,165,1,1.0,1,2,3,0
62,0,3,150,244,0,1,154,1,1.4,1,0,2,1
65,1,0,120,177,0,1,140,0,0.4,2,0,3,0
61,0,0,130,330,0,0,169,0,0.0,2,0,2,0
68,1,0,144,193,1,1,141,0,3.4,1,2,3,0
57,1,2,141,260,1,0,189,0,0.0,2,0,3,0
55,1,2,130,262,0,1,155,0,0.0,2,0,2,1
55,1,0,145,326,0,0,155,0,0.0,2,0,3,0
56,1,0,130,283,1,0,103,1,1.6,0,0,3,0
54,0,2,135,304,1,1,170,0,0.0,2,0,2,1
52,1,0,125,212,0,1,168,0,1.0,2,2,3,0
53,1,0,130,246,1,0,173,0,0.0,2,3,2,0
70,1,0,145,174,0,1,125,1,2.6,0,0,3,0
62,0,0,138,294,1,1,106,0,1.9,1,3,2,0
63,1,0,130,254,0,0,147,0,1.4,1,1,3,0
45,1,0,128,308,0,0,170,0,0.0,2,0,3,0
65,1,0,138,282,1,0,174,0,1.4,1,1,2,0
42,1,2,120,295,0,1,162,0,0.0,2,0,2,1
58,1,2,136,164,0,0,99,1,2.0,1,0,2,0
59,1,0,110,254,0,0,142,1,0.0,1,1,3,0
60,1,0,145,282,0,0,142,1,2.8,1,2,3,0
63,1,0,130,254,0,0,147,0,1.4,1,1,3,0
47,1,0,130,253,0,1,179,0,0.0,2,0,2,1
62,0,0,160,164,0,0,145,0,6.2,0,3,3,0
43,1,0,150,247,0,1,171,0,1.5,2,0,2,0
57,0,0,140,241,0,1,123,1,0.2,1,0,3,0
57,1,0,110,335,0,1,143,1,3.0,1,1,3,0
55,1,0,132,353,0,1,132,1,1.2,1,1,3,0
63,0,2,150,407,0,0,154,0,4.0,1,3,3,0
64,1,0,120,246,0,0,96,1,2.2,0,1,2,0"""

# Parse embedded data
from io import StringIO
columns = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']
df = pd.read_csv(StringIO(data_raw), header=None, names=columns)
df['target'] = (df['target'] > 0).astype(int)

print(f"\n[OK] Dataset loaded: {df.shape[0]} patients x {df.shape[1]-1} features")
print(f"   No Disease (0): {(df.target==0).sum()} | Disease (1): {(df.target==1).sum()}")

# ──────────────────────────────
#  2. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────
print("\n[STATS] Basic Statistics:")
print(df.describe().round(2))

# ──────────────────────────────
#  3. PREPROCESSING
# ──────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ──────────────────────────────
#  4. TRAIN MODELS
# ──────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
}

results = {}
print("\n[ML] Training Models...")
print("-" * 50)

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)
    acc    = accuracy_score(y_test, y_pred)
    cv     = cross_val_score(model, X_train_sc, y_train, cv=5).mean()
    results[name] = {"model": model, "acc": acc, "cv": cv, "y_pred": y_pred}
    print(f"  {name:<25}  Test Acc: {acc:.3f}  |  CV Acc: {cv:.3f}")

best_name = max(results, key=lambda k: results[k]["acc"])
print(f"\n[BEST] Best Model: {best_name} ({results[best_name]['acc']:.3f})")

# ──────────────────────────────
#  5. VISUALISATIONS
# ──────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor('#0f1117')
gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

ACCENT  = "#e84393"
BLUE    = "#4fc3f7"
GREEN   = "#69f0ae"
YELLOW  = "#ffd54f"
PALETTE = [BLUE, ACCENT]
TXTC    = "white"
BGAX    = "#1a1d2e"

def style_ax(ax, title=""):
    ax.set_facecolor(BGAX)
    ax.tick_params(colors=TXTC, labelsize=9)
    ax.xaxis.label.set_color(TXTC)
    ax.yaxis.label.set_color(TXTC)
    if title:
        ax.set_title(title, color=TXTC, fontsize=11, fontweight='bold', pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3150')

# ── Plot 1: Target distribution
ax1 = fig.add_subplot(gs[0, 0])
counts = df['target'].value_counts().sort_index()
bars = ax1.bar(['No Disease', 'Disease'], counts.values, color=PALETTE, width=0.5, edgecolor='none', zorder=3)
for b in bars:
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+2, str(int(b.get_height())),
             ha='center', color=TXTC, fontsize=11, fontweight='bold')
style_ax(ax1, "Target Distribution")
ax1.set_ylabel("Count", color=TXTC)

# ── Plot 2: Age distribution by target
ax2 = fig.add_subplot(gs[0, 1])
for tgt, color, label in zip([0,1], PALETTE, ['No Disease','Disease']):
    ax2.hist(df[df.target==tgt]['age'], bins=15, alpha=0.7, color=color, label=label, edgecolor='none')
ax2.legend(facecolor=BGAX, labelcolor=TXTC, fontsize=8)
style_ax(ax2, "Age Distribution by Outcome")
ax2.set_xlabel("Age"); ax2.set_ylabel("Count")

# ── Plot 3: Sex vs Disease
ax3 = fig.add_subplot(gs[0, 2])
sex_tab = df.groupby(['sex','target']).size().unstack(fill_value=0)
sex_tab.index = ['Female','Male']
sex_tab.columns = ['No Disease','Disease']
sex_tab.plot(kind='bar', ax=ax3, color=PALETTE, edgecolor='none', zorder=3, width=0.6)
ax3.set_xticklabels(['Female','Male'], rotation=0)
ax3.legend(facecolor=BGAX, labelcolor=TXTC, fontsize=8)
style_ax(ax3, "Sex vs Heart Disease")

# ── Plot 4: Chest pain type
ax4 = fig.add_subplot(gs[1, 0])
cp_tab = df.groupby(['cp','target']).size().unstack(fill_value=0)
cp_labels = {0:'Typical\nAngina', 1:'Atypical\nAngina', 2:'Non-anginal\nPain', 3:'Asymptomatic'}
cp_tab.index = [cp_labels.get(i, i) for i in cp_tab.index]
cp_tab.columns = ['No Disease','Disease']
cp_tab.plot(kind='bar', ax=ax4, color=PALETTE, edgecolor='none', zorder=3, width=0.7)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=15, fontsize=7)
ax4.legend(facecolor=BGAX, labelcolor=TXTC, fontsize=7)
style_ax(ax4, "Chest Pain Type vs Outcome")

# ── Plot 5: Max Heart Rate
ax5 = fig.add_subplot(gs[1, 1])
for tgt, color, label in zip([0,1], PALETTE, ['No Disease','Disease']):
    ax5.hist(df[df.target==tgt]['thalach'], bins=20, alpha=0.7, color=color, label=label, edgecolor='none')
ax5.legend(facecolor=BGAX, labelcolor=TXTC, fontsize=8)
style_ax(ax5, "Max Heart Rate by Outcome")
ax5.set_xlabel("Max Heart Rate"); ax5.set_ylabel("Count")

# ── Plot 6: Correlation heatmap
ax6 = fig.add_subplot(gs[1, 2])
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, ax=ax6, cmap='coolwarm', center=0,
            linewidths=0.5, linecolor='#0f1117', annot=False,
            cbar_kws={'shrink': 0.8})
ax6.tick_params(colors=TXTC, labelsize=6.5)
ax6.set_facecolor(BGAX)
ax6.set_title("Feature Correlation", color=TXTC, fontsize=11, fontweight='bold', pad=8)

# ── Plot 7-9: Confusion matrices
for idx, (name, res) in enumerate(results.items()):
    ax = fig.add_subplot(gs[2, idx])
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap=sns.color_palette("blend:#1a1d2e,#e84393", as_cmap=True),
                xticklabels=['No Dis.','Disease'],
                yticklabels=['No Dis.','Disease'],
                linewidths=1, linecolor='#0f1117',
                cbar=False, annot_kws={"size": 14, "weight": "bold", "color": "white"})
    ax.tick_params(colors=TXTC, labelsize=8)
    ax.set_xlabel("Predicted", color=TXTC, fontsize=9)
    ax.set_ylabel("Actual",    color=TXTC, fontsize=9)
    short = name.replace(" ", "\n")
    ax.set_title(f"{short}\nAcc: {res['acc']:.3f}", color=TXTC, fontsize=10, fontweight='bold')
    ax.set_facecolor(BGAX)

# ── Plot 10: ROC curves
ax10 = fig.add_subplot(gs[3, 0:2])
ax10.set_facecolor(BGAX)
colors_roc = [BLUE, ACCENT, GREEN]
for (name, res), color in zip(results.items(), colors_roc):
    if hasattr(res['model'], "predict_proba"):
        proba = res['model'].predict_proba(X_test_sc)[:,1]
    else:
        proba = res['model'].decision_function(X_test_sc)
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    ax10.plot(fpr, tpr, color=color, lw=2,
              label=f"{name} (AUC={roc_auc:.3f})")
ax10.plot([0,1],[0,1], color='#555', lw=1, linestyle='--')
ax10.set_xlim([0,1]); ax10.set_ylim([0,1.02])
ax10.set_xlabel("False Positive Rate", color=TXTC)
ax10.set_ylabel("True Positive Rate",  color=TXTC)
ax10.set_title("ROC Curves – All Models", color=TXTC, fontsize=12, fontweight='bold', pad=8)
ax10.tick_params(colors=TXTC)
for spine in ax10.spines.values(): spine.set_edgecolor('#2d3150')
ax10.legend(facecolor=BGAX, labelcolor=TXTC, fontsize=9)

# ── Plot 11: Feature importances (RF)
ax11 = fig.add_subplot(gs[3, 2])
rf = results["Random Forest"]["model"]
feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
colors_fi = [ACCENT if v > feat_imp.median() else BLUE for v in feat_imp]
ax11.barh(feat_imp.index, feat_imp.values, color=colors_fi, edgecolor='none')
style_ax(ax11, "Feature Importance (RF)")
ax11.set_xlabel("Importance", color=TXTC)

# Title
fig.suptitle("Heart Disease Prediction Dashboard | UCI Cleveland Dataset",
             color=TXTC, fontsize=16, fontweight='bold', y=0.995)

plt.savefig('heart_disease_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
print("\n[OK] Dashboard saved -> heart_disease_dashboard.png")

# -- Save CSV dataset
df.to_csv('heart_disease_dataset.csv', index=False)
print("[OK] Dataset saved -> heart_disease_dataset.csv")

# ── Final Report
print("\n" + "="*60)
print("  FINAL RESULTS SUMMARY")
print("="*60)
for name, res in results.items():
    print(f"\n  {name}")
    print(f"  Test Accuracy : {res['acc']:.4f}")
    print(f"  5-Fold CV Acc : {res['cv']:.4f}")
    report = classification_report(y_test, res['y_pred'],
             target_names=['No Disease','Disease'])
    print(report)

# ── DEMO: Predict for a new patient
print("="*60)
print("  DEMO PREDICTION")
print("="*60)
sample = np.array([[52, 1, 0, 125, 212, 0, 1, 168, 0, 1.0, 2, 2, 3]])
sample_sc = scaler.transform(sample)
best_model = results[best_name]["model"]
pred = best_model.predict(sample_sc)[0]
prob = best_model.predict_proba(sample_sc)[0]
print(f"\n  Patient: 52yr Male | Chest Pain Type 0 | Chol=212 | HR=168")
print(f"  Prediction : {'[WARNING] DISEASE PRESENT' if pred==1 else '[OK] No Disease'}")
print(f"  Confidence : No Disease={prob[0]:.2%} | Disease={prob[1]:.2%}")
print(f"  (Model: {best_name})")
print("\n[OK] Done!")
