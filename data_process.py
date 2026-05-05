import pandas as pd


# dataset here: https://www.kaggle.com/datasets/ikynahidwin/depression-student-dataset?resource=download

df = pd.read_csv("Data/Depression Student Dataset.csv")
col = df.columns.tolist()

print(col)

# data processing

df[col[0]] = df[col[0]].map({"Male": 0, "Female": 1}).astype(int)
df[col[1]] = df[col[1]].astype(int)
df[col[2]] = df[col[2]].astype(float)
df[col[3]] = df[col[3]].astype(float)
df[col[5]] = df[col[5]].map({"Healthy": 1, "Moderate": 0.5, "Unhealthy": 0}).astype(float)
df[col[6]] = df[col[6]].map({"Yes": 1, "No": 0}).astype(int)
df[col[7]] = df[col[7]].astype(int)
df[col[8]] = df[col[8]].astype(int)
df[col[9]] = df[col[9]].map({"Yes": 1, "No": 0}).astype(int)
df[col[10]] = df[col[10]].map({"Yes": 1, "No": 0}).astype(int)

for i, item in enumerate(df[col[4]]):
    if "Less" in item:
        df.iloc[i, 4] = "4.5"
    elif "More" in item:
        df.iloc[i, 4] = "8.5"
    elif "-" in item:
        df.iloc[i, 4] = str(abs(float(item[2]) + float(item[0])) / 2)
    else:
        raise ValueError(f"unknown value in {col[4]}")
    # assigning numerical values to sleep hours
df[col[4]].astype(float)


df.to_csv("Data/Data_processed.csv", index=False)