import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

file_ = "MetObjects.csv"
CHUNK_SIZE = 10000
OUTPUT_DIR = "results"

def read_chunks(file_path):
    for chunk in pd.read_csv(file_path, chunksize=CHUNK_SIZE):
        yield chunk

def filter_data(chunks):
    for df in chunks:
        df = df[["Country", "Object Begin Date"]]
        df = df.dropna()
        df["Century"] = (df["Object Begin Date"] // 100) + 1
        yield df

def aggregate(chunks):
    total_df = pd.DataFrame()

    for df in chunks:
        total_df = pd.concat([total_df, df])

    top_countries = total_df["Country"].value_counts().head(10).index
    df_top = total_df[total_df["Country"].isin(top_countries)]

    grouped = df_top.groupby("Country")["Century"]

    result = pd.DataFrame()
    result["mean"] = grouped.mean()
    result["std"] = grouped.std()
    result["count"] = grouped.count()

    def calc_ci(row):
        n = row["count"]
        ci = stats.t.interval(0.95, df=n - 1, loc=row["mean"], scale=row["std"] / np.sqrt(n))
        return pd.Series({'ci_lower': ci[0], 'ci_upper': ci[1]})
    result[['ci_lower', 'ci_upper']] = result.apply(calc_ci, axis=1)

    def calc_scatter(group):
        return pd.Series({
            'scatter_lower': group.quantile(0.025),
            'scatter_upper': group.quantile(0.975)
        })
    scatter = df_top.groupby('Country')['Century'].apply(calc_scatter).unstack()
    result['scatter_lower'] = scatter['scatter_lower']
    result['scatter_upper'] = scatter['scatter_upper']

    result = result.reset_index()  # сбрасывает индекс
    print("Результаты (топ-10 стран)")
    result["ci95"]=result['ci_upper']-result["ci_lower"]
    print(result[["Country", "mean", "ci_lower","ci_upper","ci95", "scatter_lower", "scatter_upper",  "count"]].round(2).to_string(index=False))
    return result, df_top
def plot_bar(result):
    result = result.sort_values("mean", ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12, 6))
    x = range(len(result))
    plt.bar(x, result["mean"], color="pink",  edgecolor="blue")
    for i, row in result.iterrows():
        plt.plot([i, i], [row["ci_lower"], row["ci_upper"]], 'b-', linewidth=3)
        plt.plot(i, row["ci_lower"], 'bv', markersize=6)
        plt.plot(i, row["ci_upper"], 'b^', markersize=6)
    plt.title("Средний век по странам (топ-10)\nс 95% доверительным интервалом", fontsize=14)
    plt.ylabel("Век")
    plt.xlabel("Страна")
    plt.xticks(x, result["Country"], rotation=45, ha="right")
    plt.grid(alpha=0.3)
    bar_path = os.path.join(OUTPUT_DIR, "bar_chart.png")
    plt.savefig(bar_path)
    print(f"Столбчатая диаграмма сохранена: {bar_path}")
    plt.show()

def plot_time(df):
    country_ = df.groupby("Country")["Century"].mean().idxmax()
    print(f"\nСамая современная страна: {country_}")

    df_country = df[df["Country"] == country_]
    counts = df_country.groupby("Object Begin Date").size().sort_index()

    window = max(5, len(counts) // 10)
    rolling = counts.rolling(window=window, min_periods=1).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(counts.index, counts.values, "b-", alpha=0.5, label="Количество объектов")
    plt.plot(rolling.index, rolling.values, "g-", label=f"Скользящее среднее (окно={window})")
    plt.title(f"Динамика создания объектов: {country_}")
    plt.xlabel("Год")
    plt.ylabel("Количество объектов")
    plt.legend()
    plt.grid(alpha=0.3)
    ts_path = os.path.join(OUTPUT_DIR, "time_series.png")
    plt.savefig(ts_path, dpi=300, bbox_inches="tight")
    print(f"Временной график сохранён: {ts_path}")
    plt.show()



def main_pipeline():
    result, df_top = aggregate(filter_data(read_chunks(file_)))
    plot_bar(result)
    plot_time(df_top)
    print(f"\nВсе графики сохранены в папку: {OUTPUT_DIR}")

if __name__ == "__main__":
    main_pipeline()