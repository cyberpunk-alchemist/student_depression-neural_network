import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Data/Data_processed.csv")
col = df.columns.tolist()

#bar charts
for feature in col:
    fig, ax = plt.subplots()
    df.boxplot(column=feature, ax=ax)
    if feature[-1] == "?":
        fig.savefig(f"graphs/{feature[:-3]}-boxplot.png", dpi=300, bbox_inches="tight")
    else:
        fig.savefig(f"graphs/{feature}-boxplot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots()
    df.hist(column=feature, ax=ax,bins=15)
    if feature[-1] == "?":
        fig.savefig(f"graphs/{feature[:-3]}-hist.png", dpi=300, bbox_inches="tight")
    else:
        fig.savefig(f"graphs/{feature}-hist.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


corr_matrix = df.corr(numeric_only=True)
#corelation matrix
labels=[]
for name in col:
    labels.append(name[0:5])

fig, ax = plt.subplots()
im = ax.imshow(corr_matrix, aspect="auto",  cmap="Reds")
fig.colorbar(im, ax=ax)
ax.set_xticks(np.arange(len(col)))
ax.set_yticks(np.arange(len(col)))
ax.set_xticklabels(labels, fontsize=8, rotation=90)
ax.set_yticklabels(labels, fontsize=8)
ax.set_title("Correlation matrix")

for i in range(corr_matrix.shape[0]):
    for j in range(corr_matrix.shape[1]):
        ax.text(j, i, f"{corr_matrix.iloc[i,j]:.3f}", ha="center", va="center",fontsize=7)

plt.tight_layout()
plt.savefig("graphs/correlation.png", dpi=300, bbox_inches="tight")
plt.close()


