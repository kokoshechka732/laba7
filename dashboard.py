import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "DejaVu Sans"

global cdr

data = np.load("data_final.npy", allow_pickle=True)
df_raw = pd.DataFrame(data)

# Глобальный ID поста
df_raw["post_id"] = range(1, len(df_raw) + 1)

df_raw["ts"] = pd.date_range(
    start="2026-05-01",
    end="2026-05-31 23:59:59",
    periods=len(df_raw)
)

df_raw["ts"] = pd.to_datetime(df_raw["ts"])
df_raw["acc_id"] = df_raw["acc_id"].astype(str)


#Признаки

def updatec(dataf):
    dataf = dataf.copy()

    dataf["engagement"] = (
        dataf.get("likes", 0)
        + dataf.get("shares", 0)
        + dataf.get("com", 0)
    )

    dataf = dataf.sort_values(["acc_id", "ts"])

    dataf["virality"] = (
        dataf.groupby("acc_id")["shares"]
        .diff()
        .fillna(0)
    )

    dataf["engagement_rate"] = dataf["engagement"] / (dataf["engagement"].max() + 1)

    high_eng = dataf["engagement"].quantile(0.75)
    high_vir = dataf["virality"].quantile(0.75)

    conditions = [
        (dataf["engagement"] > high_eng) & (dataf["virality"] > high_vir),
        (dataf["engagement"] > high_eng)
    ]

    choices = [
        "Вирусный",
        "Интересный"
    ]

    dataf["post_quality"] = np.select(
        conditions,
        choices,
        default="Обычный"
    )

    return dataf

dataf_work = updatec(df_raw)


# TKINTER — ОКНО

root = tk.Tk()
root.title("Дашборд: Анализ активности")
root.geometry("1300x850")

main = tk.Frame(root)
main.pack(fill=tk.BOTH, expand=True)

left = tk.Frame(main, width=250)
left.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

right = tk.Frame(main)
right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

fig = plt.figure(figsize=(9, 5), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=right)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

toolbar = NavigationToolbar2Tk(canvas, right)
toolbar.update()
toolbar.pack(side=tk.TOP, fill=tk.X)


# ПЕРЕМЕННЫЕ

acc_var     = tk.StringVar(value="Все")
metric_var  = tk.StringVar(value="engagement")
agg_var     = tk.StringVar(value="sum")
quality_var = tk.StringVar(value="Все")
cur_chart   = "line"


# ФИЛЬТР

def get_filtered():
    dataf = dataf_work.copy()   # берём уже обработанный датафрейм

    acc = acc_var.get()
    if acc != "Все":
        dataf = dataf[dataf["acc_id"] == acc]

    post_q = quality_var.get()
    if post_q != "Все":
        dataf = dataf[dataf["post_quality"] == post_q]

    return dataf


# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ

def clear():
    fig.clear()

def show_empty():
    clear()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, "Нет данных",
            ha="center", va="center", fontsize=14)
    canvas.draw_idle()


# ПОСТРОЕНИЕ ГРАФИКОВ

def plot_line():
    global cur_chart
    cur_chart = "line"

    dataf = get_filtered()
    if dataf.empty:
        show_empty()
        return

    clear()

    metric = metric_var.get()
    agg    = agg_var.get()

    dataf_daily = (
        dataf.set_index("ts")
        .resample("D")[metric]
        .agg(agg)
        .reset_index()
    )

    ax = fig.add_subplot(111)
    sns.lineplot(data=dataf_daily, x="ts", y=metric, ax=ax)

    ax.set_title(f"{metric} по дням (Май 2026)")
    ax.set_xlabel("Дата")
    ax.set_ylabel(metric)

    fig.tight_layout()
    canvas.draw_idle()


def plot_bar():
    global cur_chart
    cur_chart = "bar"

    dataf = get_filtered()
    if dataf.empty:
        show_empty()
        return

    clear()

    metric = metric_var.get()
    agg    = agg_var.get()

    grouped = (
        dataf.groupby("acc_id")[metric]
        .agg(agg)
        .reset_index()
        .sort_values(metric, ascending=False)
        .head(10)
    )

    ax = fig.add_subplot(111)
    sns.barplot(data=grouped, x="acc_id", y=metric, ax=ax)

    ax.set_title(f"Топ аккаунтов по {metric}")
    fig.tight_layout()
    canvas.draw_idle()


def plot_scatter():
    global cur_chart
    cur_chart = "scatter"

    dataf = get_filtered()
    if dataf.empty:
        show_empty()
        return

    clear()

    palette = {
        "Обычный":    "gray",
        "Интересный": "orange",
        "Вирусный":   "red"
    }

    ax = fig.add_subplot(111)
    sns.scatterplot(
        data=dataf,
        x="post_id",
        y="likes",
        hue="post_quality",
        palette=palette,
        alpha=0.6,
        ax=ax
    )

    ax.set_title("Зависимость лайков от ID поста")
    ax.set_xlabel("Глобальный ID поста")
    ax.set_ylabel("Количество лайков")

    fig.tight_layout()
    canvas.draw_idle()


def plot_heatmap():
    global cur_chart
    cur_chart = "heat"

    dataf = get_filtered()
    if dataf.empty:
        show_empty()
        return

    clear()

    corr = dataf[[
        "likes",
        "shares",
        "com",
        "engagement",
        "virality",
        "engagement_rate"
    ]].corr()

    ax = fig.add_subplot(111)
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)

    ax.set_title("Корреляция признаков")
    fig.tight_layout()
    canvas.draw_idle()


# ОБНОВЛЕНИЕ ГРАФИКА

def update_chart(event=None):
    if cur_chart == "line":
        plot_line()
    elif cur_chart == "bar":
        plot_bar()
    elif cur_chart == "scatter":
        plot_scatter()
    elif cur_chart == "heat":
        plot_heatmap()


# GUI — ЛЕВАЯ ПАНЕЛЬ


# Аккаунт
tk.Label(left, text="Аккаунт").pack(pady=5)
acc_list  = ["Все"] + sorted(df_raw["acc_id"].unique())
acc_combo = ttk.Combobox(left, textvariable=acc_var, values=acc_list)
acc_combo.pack(pady=5)
acc_combo.bind("<<ComboboxSelected>>", update_chart)

# Метрика
tk.Label(left, text="Метрика").pack(pady=5)
metric_combo = ttk.Combobox(
    left,
    textvariable=metric_var,
    values=["engagement", "likes", "shares", "com"]
)
metric_combo.pack(pady=5)
metric_combo.bind("<<ComboboxSelected>>", update_chart)

# Агрегация
tk.Label(left, text="Агрегация").pack(pady=5)
tk.Radiobutton(left, text="Sum",    variable=agg_var, value="sum",    command=update_chart).pack()
tk.Radiobutton(left, text="Mean",   variable=agg_var, value="mean",   command=update_chart).pack()
tk.Radiobutton(left, text="Median", variable=agg_var, value="median", command=update_chart).pack()

# Качество поста
tk.Label(left, text="Качество поста").pack(pady=5)
quality_combo = ttk.Combobox(
    left,
    textvariable=quality_var,
    values=["Все", "Обычный", "Интересный", "Вирусный"]
)
quality_combo.pack(pady=5)
quality_combo.bind("<<ComboboxSelected>>", update_chart)

# Кнопки графиков
tk.Button(left, text="Line",     command=plot_line).pack(fill=tk.X, pady=5)
tk.Button(left, text="Bar",      command=plot_bar).pack(fill=tk.X, pady=5)
tk.Button(left, text="Scatter",  command=plot_scatter).pack(fill=tk.X, pady=5)
tk.Button(left, text="Heatmap",  command=plot_heatmap).pack(fill=tk.X, pady=5)

# Экспорт
def export_plot():
    path = filedialog.asksaveasfilename(defaultextension=".png")
    if path:
        fig.savefig(path, dpi=300, bbox_inches="tight")
        messagebox.showinfo("Экспорт", "Сохранено")

tk.Button(left, text="Экспорт", command=export_plot).pack(fill=tk.X, pady=20)


# ЗАПУСК

np.save("data_final33.npy", dataf_work)

print(dataf_work.columns.tolist())
print(dataf_work.dtypes)

plot_line()
root.mainloop()