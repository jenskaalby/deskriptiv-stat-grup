import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide")
st.title("📘 Sumkurve og deskriptiv statistik for grupperede data")

# Sidepanel med licens og udvikler
with st.sidebar:
    st.markdown("### ℹ️ Om")
    st.markdown("""
    **Licens:** MIT License  
    **Udvikler:** Jens Kaalby Thomsen 
    Dette værktøj hjælper med at visualisere og beregne deskriptiv statistik for grupperede observationer.
    """)

# INPUT
st.header("1. Indtast intervaller og hyppigheder")
col1, col2 = st.columns(2)
with col1:
    interval_input = st.text_area("Intervaller (et pr. linje, fx '10-20'):", height=200, key="interval_input")
with col2:
    freq_input = st.text_area("Hyppigheder (en pr. linje, fx '4'):", height=200, key="freq_input")

if st.button("🔄 Ryd inputs"):
    st.experimental_rerun()

if interval_input and freq_input:
    try:
        interval_lines = interval_input.strip().split("\n")
        freq_lines = freq_input.strip().split("\n")

        if len(interval_lines) != len(freq_lines):
            st.error("Antallet af intervaller og hyppigheder skal være det samme.")
        else:
            # Parse inputs
            intervals = []
            for line in interval_lines:
                a, b = map(float, line.strip().split("-"))
                intervals.append((a, b))
            frequencies = list(map(int, freq_lines))

            N = sum(frequencies)
            midpoints = [(a + b) / 2 for a, b in intervals]
            mean = sum(f * x for f, x in zip(frequencies, midpoints)) / N

            # Varians og spredning (population og stikprøve)
            variance_population = sum(f * (x - mean) ** 2 for f, x in zip(frequencies, midpoints)) / N
            std_dev_population = np.sqrt(variance_population)
            variance_sample = sum(f * (x - mean) ** 2 for f, x in zip(frequencies, midpoints)) / (N - 1) if N > 1 else float("nan")
            std_dev_sample = np.sqrt(variance_sample)

            # DataFrame med omdøbte kolonner og fjernet indeks
            df = pd.DataFrame({
                "Interval": [f"{a}-{b}" for a, b in intervals],
                "Hyppighed": frequencies,
                "Midtpunkter": midpoints
            })
            df["Kumulativ hyppighed"] = np.cumsum(df["Hyppighed"])
            df["Kumulativ frekvens"] = df["Kumulativ hyppighed"] / N

            st.subheader("📊 Oversigt over data")
            st.table(df.reset_index(drop=True))

            # Fraktil input
            st.subheader("📍 Fraktilberegning")
            col3, col4 = st.columns(2)
            with col3:
                p_input = st.number_input("Indtast fraktil (fx 0.25):", min_value=0.0, max_value=1.0, step=0.01, key="fraktil")
            with col4:
                x_input = st.number_input("Indtast værdi for at finde fraktil:", step=0.1, key="værdi")

            ekstra_fraktiler = st.multiselect("Vælg ekstra fraktiler (p):", [0.25, 0.5, 0.75, 0.9])

            fraktilpunkt = None
            værdi_punkt = None
            ekstra_punkter = []

            if p_input > 0:
                target = p_input * N
                cum_freq = 0
                for i, freq in enumerate(frequencies):
                    cum_freq += freq
                    if cum_freq >= target:
                        a, b = intervals[i]
                        L = a
                        h = b - a
                        F = sum(frequencies[:i])
                        f_i = frequencies[i]
                        fraktil_værdi = L + ((target - F) / f_i) * h
                        st.write(f"🎯 **Fraktilværdi for p = {p_input}** er **{fraktil_værdi:.2f}**")
                        fraktilpunkt = (fraktil_værdi, p_input)
                        break

            if x_input > 0:
                for i, (a, b) in enumerate(intervals):
                    if x_input <= b:
                        L = a
                        h = b - a
                        F = sum(frequencies[:i])
                        f_i = frequencies[i]
                        p_beregnet = (F + f_i * (x_input - L) / h) / N
                        st.write(f"🎯 **Fraktil p for x = {x_input}** er **{p_beregnet:.2f}**")
                        værdi_punkt = (x_input, p_beregnet)
                        break

            for p_valg in ekstra_fraktiler:
                target = p_valg * N
                cum_freq = 0
                for i, freq in enumerate(frequencies):
                    cum_freq += freq
                    if cum_freq >= target:
                        a, b = intervals[i]
                        L = a
                        h = b - a
                        F = sum(frequencies[:i])
                        f_i = frequencies[i]
                        fraktil_værdi = L + ((target - F) / f_i) * h
                        ekstra_punkter.append((fraktil_værdi, p_valg))
                        break

            # Valg for y-akse på sumkurve
            y_val_type_sum = st.radio("Sumkurve y-akse som:", ("Hyppighed", "Frekvens (%)"), horizontal=True, key="sum_radio")
            # Valg for y-akse på histogram
            y_val_type_hist = st.radio("Histogram y-akse som:", ("Hyppighed", "Frekvens (%)"), horizontal=True, key="hist_radio")

            # --- Sumkurve plot ---
            st.subheader("📈 Sumkurve")
            fig_sum, ax_sum = plt.subplots(figsize=(4, 2.5))

            x_vals = [intervals[0][0]] + [b for (_, b) in intervals]
            if y_val_type_sum == "Hyppighed":
                y_vals = [0.0] + list(df["Kumulativ hyppighed"])
                ylabel_sum = "Kumulativ hyppighed"
            else:
                y_vals = [0.0] + list(df["Kumulativ frekvens"] * 100)
                ylabel_sum = "Kumulativ frekvens (%)"
            ax_sum.plot(x_vals, y_vals, marker="o", label="Sumkurve")

            def tegn_markering(ax, x, y, color, label=None):
                ax.axvline(x=x, ymin=0, ymax=y / max(y_vals) if max(y_vals) != 0 else 1, color=color, linestyle="dotted")
                ax.axhline(y=y, xmax=(x - x_vals[0]) / (x_vals[-1] - x_vals[0]), color=color, linestyle="dotted")
                ax.plot(x, y, "o", color=color)
                if label:
                    ax.text(x, y, f"  {label}", verticalalignment="bottom", fontsize=9)

            if fraktilpunkt:
                x, p = fraktilpunkt
                y = p * N if y_val_type_sum == "Hyppighed" else p * 100
                tegn_markering(ax_sum, x, y, "red", f"p={p:.2f}")

            if værdi_punkt:
                x, p = værdi_punkt
                y = p * N if y_val_type_sum == "Hyppighed" else p * 100
                tegn_markering(ax_sum, x, y, "blue", f"x={x:.2f}")

            for x, p in ekstra_punkter:
                y = p * N if y_val_type_sum == "Hyppighed" else p * 100
                tegn_markering(ax_sum, x, y, "green", f"p={p:.2f}")

            ax_sum.set_xlabel("Værdi / intervalgrænse")
            ax_sum.set_ylabel(ylabel_sum)
            ax_sum.set_title("Sumkurve")
            ax_sum.grid(True)
            ax_sum.legend()
            st.pyplot(fig_sum, use_container_width=False)

            # Download sumkurve som PNG eller SVG
            buf_sum_png = BytesIO()
            fig_sum.savefig(buf_sum_png, format="png")
            buf_sum_png.seek(0)

            buf_sum_svg = BytesIO()
            fig_sum.savefig(buf_sum_svg, format="svg")
            buf_sum_svg.seek(0)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="Download sumkurve PNG",
                    data=buf_sum_png,
                    file_name="sumkurve.png",
                    mime="image/png"
                )
            with col_dl2:
                st.download_button(
                    label="Download sumkurve SVG",
                    data=buf_sum_svg,
                    file_name="sumkurve.svg",
                    mime="image/svg+xml"
                )

            # --- Histogram plot ---
            st.subheader("📊 Histogram")

            fig_hist, ax_hist = plt.subplots(figsize=(4, 2.5))
            widths = [b - a for a, b in intervals]

            if y_val_type_hist == "Hyppighed":
                heights = frequencies
                ylabel_hist = "Hyppighed"
            else:
                heights = [(f / N) * 100 for f in frequencies]
                ylabel_hist = "Frekvens (%)"

            # Histogram - ikke centrerede søjler, start ved interval venstre grænse
            ax_hist.bar([a for a, _ in intervals], heights, width=widths, align='edge', edgecolor='black')

            ax_hist.set_xlabel("Interval")
            ax_hist.set_ylabel(ylabel_hist)
            ax_hist.set_title("Histogram")
            ax_hist.grid(axis='y')
            st.pyplot(fig_hist, use_container_width=False)

            # Download histogram som PNG eller SVG
            buf_hist_png = BytesIO()
            fig_hist.savefig(buf_hist_png, format="png")
            buf_hist_png.seek(0)

            buf_hist_svg = BytesIO()
            fig_hist.savefig(buf_hist_svg, format="svg")
            buf_hist_svg.seek(0)

            col_dl3, col_dl4 = st.columns(2)
            with col_dl3:
                st.download_button(
                    label="Download histogram PNG",
                    data=buf_hist_png,
                    file_name="histogram.png",
                    mime="image/png"
                )
            with col_dl4:
                st.download_button(
                    label="Download histogram SVG",
                    data=buf_hist_svg,
                    file_name="histogram.svg",
                    mime="image/svg+xml"
                )

            # --- Statistik visning ---
            st.subheader("📋 Statistik")

            statistik_data = {
                "Parameter": ["n", "Middelværdi", "Populationsvarians", "Populationsspredning",
                              "Stikprøvevarians", "Stikprøvespredning"],
                "Værdi": [N, mean, variance_population, std_dev_population, variance_sample, std_dev_sample]
            }
            df_stat = pd.DataFrame(statistik_data)
            df_stat["Værdi"] = df_stat["Værdi"].apply(lambda x: f"{x:.2f}")
            st.table(df_stat)

    except Exception as e:
        st.error(f"Fejl i input eller beregning: {e}")

else:
    st.info("Indtast intervaller og hyppigheder for at starte beregningen.")
