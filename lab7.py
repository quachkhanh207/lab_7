import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# =================================================================
# CHUẨN BỊ DỮ LIỆU
# =================================================================
df = pd.read_csv('ITA105_Lab_7.csv')
num_cols = df.select_dtypes(include=[np.number]).columns

# =================================================================
# BÀI 1: PHÂN TÍCH DỮ LIỆU & KHÁM PHÁ PHÂN PHỐI
# =================================================================
print("--- BÀI 1: PHÂN TÍCH SKEWNESS ---")
# Tính skewness cho toàn bộ cột số
skew_values = df[num_cols].skew().sort_values(ascending=False)
print("Bảng thứ hạng top 10 cột lệch nhất:\n", skew_values.head(10))

# Vẽ biểu đồ cho 3 cột lệch mạnh nhất (Top 2 dương và 1 âm cuối bảng)
top_cols = [skew_values.index[0], skew_values.index[1], skew_values.index[-1]]

plt.figure(figsize=(18, 5))
for i, col in enumerate(top_cols):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True, color='teal')
    plt.title(f'Phân phối của {col}\nSkew: {df[col].skew():.2f}')
plt.tight_layout()
plt.show()

# =================================================================
# BÀI 2: BIẾN ĐỔI DỮ LIỆU NÂNG CAO
# =================================================================
print("\n--- BÀI 2: BIẾN ĐỔI DỮ LIỆU ---")
# 1. np.log() cho cột dương (SalePrice)
df['SalePrice_log'] = np.log(df['SalePrice'])

# 2. boxcox() cho cột dương (LotArea)
df['LotArea_boxcox'], lmbda = stats.boxcox(df['LotArea'])
print(f"Lambda tối ưu cho LotArea: {lmbda:.4f}")

# 3. PowerTransformer (Yeo-Johnson) cho cột có giá trị âm (NegSkewIncome)
pt = PowerTransformer(method='yeo-johnson')
df['NegSkewIncome_pt'] = pt.fit_transform(df[['NegSkewIncome']])

# Lập bảng so sánh
comparison = pd.DataFrame({
    'Cột': ['SalePrice', 'LotArea', 'NegSkewIncome'],
    'Skew Gốc': [df['SalePrice'].skew(), df['LotArea'].skew(), df['NegSkewIncome'].skew()],
    'Skew Sau Transform': [df['SalePrice_log'].skew(), df['LotArea_boxcox'].skew(), df['NegSkewIncome_pt'].skew()],
    'Phương Pháp': ['Log', 'Box-Cox', 'Yeo-Johnson']
})
print(comparison)

# =================================================================
# BÀI 3: ỨNG DỤNG VÀO MÔ HÌNH HÓA (LINEAR REGRESSION)
# =================================================================
print("\n--- BÀI 3: HUẤN LUYỆN MÔ HÌNH ---")
# Chọn đặc trưng (Features) và Biến mục tiêu (Target)
features = ['LotArea', 'HouseAge', 'Rooms']
X = df[features]
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Version A: Dữ liệu gốc ---
model_a = LinearRegression()
model_a.fit(X_train, y_train)
pred_a = model_a.predict(X_test)
rmse_a = np.sqrt(mean_squared_error(y_test, pred_a))
r2_a = r2_score(y_test, pred_a)

# --- Version B: Biến đổi log biến mục tiêu ---
y_train_log = np.log(y_train)
model_b = LinearRegression()
model_b.fit(X_train, y_train_log)
pred_log_b = model_b.predict(X_test)
pred_b = np.exp(pred_log_b) # Dịch ngược về giá trị thực
rmse_b = np.sqrt(mean_squared_error(y_test, pred_b))
r2_b = r2_score(y_test, pred_b)

print(f"Version A (Gốc) - RMSE: {rmse_a:,.2f}, R2: {r2_a:.4f}")
print(f"Version B (Log) - RMSE: {rmse_b:,.2f}, R2: {r2_b:.4f}")

# =================================================================
# BÀI 4: ỨNG DỤNG NGHIỆP VỤ & RA QUYẾT ĐỊNH
# =================================================================
# Tạo biểu đồ so sánh cho insight
plt.figure(figsize=(14, 6))

# Trước khi transform
plt.subplot(1, 2, 1)
sns.scatterplot(x=df['LotArea'], y=df['SalePrice'], alpha=0.5)
plt.title("Mối quan hệ Gốc (Raw Data)\nKhó quan sát do Outliers")
plt.xlabel("Diện tích đất (LotArea)")
plt.ylabel("Giá bán (SalePrice)")

# Sau khi transform
plt.subplot(1, 2, 2)
sns.scatterplot(x=df['LotArea_boxcox'], y=df['SalePrice_log'], color='orange', alpha=0.5)
plt.title("Mối quan hệ sau Transform\nDữ liệu tập trung, rõ xu hướng")
plt.xlabel("LotArea (Box-Cox)")
plt.ylabel("SalePrice (Log)")

plt.tight_layout()
plt.show()

print("\n--- INSIGHT NGHIỆP VỤ ---")
print("1. Tại sao cần biến đổi: Dữ liệu bất động sản thường bị lệch do một số ít căn nhà có diện tích hoặc giá cực lớn.")
print("2. Tác dụng: Biến đổi giúp thu nhỏ khoảng cách giữa các giá trị cực lớn, đưa chúng về gần trung tâm hơn.")
print("3. Quyết định: Giúp nhà đầu tư nhìn thấy xu hướng tăng giá ổn định thay vì bị nhiễu bởi các căn 'biệt thự' cá biệt.")