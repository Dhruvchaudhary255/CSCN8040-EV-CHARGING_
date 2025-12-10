import os
import pandas as pd
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = "DataSets/DataSets"

# ---------- Helpers ---------- #

@st.cache_data
def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    try:
        df = pd.read_csv(path, low_memory=False)
        return df
    except FileNotFoundError:
        return None


def kpi_card(label, value):
    st.metric(label, f"{value:,}")


# ---------- Load data ---------- #

stations_df = load_csv("stations_detail.csv")
ports_df = load_csv("ports_detail.csv")
plugs_df = load_csv("plugs_detail.csv")
summary_df = load_csv("ev_city_station_summary.csv")
cities_df = load_csv("canadacities.csv")  # optional, if exists

# ---------- Sidebar navigation ---------- #

st.sidebar.title("EV Charging Dashboard")
page = st.sidebar.radio(
    "Go to",
    [
        "1️⃣ Overview",
        "2️⃣ City EV vs Charging",
        "3️⃣ Plug / Port Mix",
        "4️⃣ Station Map",
        "5️⃣ Data Explorer",
        "6️⃣ Rural Demand Index (RDI) Model",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("Data source: local CSVs in `DataSets/DataSets`")

# Common province filter (if available anywhere)
province_options = None
for df in [summary_df, stations_df, cities_df]:
    if df is not None:
        for col in df.columns:
            if col.lower() in ["province", "prov", "state"]:
                province_options = sorted(df[col].dropna().unique())
                break
    if province_options:
        break

if province_options:
    selected_province = st.sidebar.selectbox(
        "Filter by Province (optional)", ["All"] + list(province_options)
    )
else:
    selected_province = "All"


def apply_province_filter(df):
    if df is None or province_options is None or selected_province == "All":
        return df
    # find province column
    prov_col = [c for c in df.columns if c.lower() in ["province", "prov", "state"]]
    if not prov_col:
        return df
    prov_col = prov_col[0]
    return df[df[prov_col] == selected_province]


# ---------- PAGE 1: Overview ---------- #

if page.startswith("1️⃣"):
    st.title("EV Charging Access – Overview")

    st.write(
        """
        This dashboard summarizes EV charging infrastructure using multiple datasets:

        - **Stations detail** – individual charging stations  
        - **Ports / connectors** – technical details of charging ports  
        - **EV vs Stations summary** – city-level counts of EVs vs charging stations  
        """
    )

    col1, col2, col3 = st.columns(3)

    # Stations KPI
    if stations_df is not None:
        s_df = apply_province_filter(stations_df)
        kpi_card("Total Stations", len(s_df))
    else:
        col1.write("No `stations_detail.csv` found")

    # Ports KPI
    if ports_df is not None:
        p_df = apply_province_filter(ports_df)
        kpi_card("Total Ports", len(p_df))
    else:
        col2.write("No `ports_detail.csv` found")

    # Cities KPI
    if summary_df is not None:
        sum_df = apply_province_filter(summary_df)
        if "City" in sum_df.columns:
            kpi_card("Cities in Summary", sum_df["City"].nunique())
        else:
            kpi_card("Rows in Summary", len(sum_df))
    else:
        col3.write("No `ev_city_station_summary.csv` found")

    st.markdown("---")

    st.subheader("Sample of Each Dataset")

    if stations_df is not None:
        st.write("**Stations detail**")
        st.dataframe(apply_province_filter(stations_df).head())
    else:
        st.info("`stations_detail.csv` not available.")

    if ports_df is not None:
        st.write("**Ports detail**")
        st.dataframe(apply_province_filter(ports_df).head())

    if plugs_df is not None:
        st.write("**Plugs detail**")
        st.dataframe(apply_province_filter(plugs_df).head())

    if summary_df is not None:
        st.write("**EV vs Stations summary**")
        st.dataframe(apply_province_filter(summary_df).head())


# ---------- PAGE 2: City EV vs Charging ---------- #

elif page.startswith("2️⃣"):
    st.title("City-Level EV vs Charging Capacity")

    if summary_df is None:
        st.error("`ev_city_station_summary.csv` not found.")
    else:
        df = apply_province_filter(summary_df).copy()

        # Try to identify columns
        # Expect something like: Province, City, EV_Count, Charging_Stations
        # Flexible matching by lowercase name
        def find_col(options):
            for c in df.columns:
                if c.lower() in options:
                    return c
            return None

        city_col = find_col(["city"])
        prov_col = find_col(["province", "prov", "state"])
        ev_col = find_col(["ev_count", "evs", "evs_count", "no_of_evs"])
        stations_col = find_col(
            ["charging_stations", "stations", "num_stations", "no_of_stations"]
        )

        if not (city_col and ev_col and stations_col):
            st.error(
                "Could not automatically detect columns for city, EV count, and charging stations. "
                "Please check your `ev_city_station_summary.csv` column names."
            )
            st.write("Detected columns:", list(df.columns))
        else:
            st.write(
                f"Using columns: **City:** `{city_col}`, **EV Count:** `{ev_col}`, "
                f"**Charging Stations:** `{stations_col}`"
            )

            # Compute ratios / gap score
            df = df[df[stations_col] > 0].copy()
            df["EV_per_station"] = df[ev_col] / df[stations_col]

            st.subheader("Scatter: EV Count vs Charging Stations")
            fig = px.scatter(
                df,
                x=ev_col,
                y=stations_col,
                color=prov_col if prov_col else city_col,
                hover_data=[city_col],
                trendline="ols",
                labels={
                    ev_col: "EV Count",
                    stations_col: "Charging Stations",
                },
                height=500,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.subheader("Top Underserved Cities (highest EV per station)")

            top_n = st.slider("Show top N cities", 5, 50, 10)
            top_gap = df.sort_values("EV_per_station", ascending=False).head(top_n)

            st.dataframe(
                top_gap[[city_col, ev_col, stations_col, "EV_per_station"]]
                .reset_index(drop=True)
            )

            fig2 = px.bar(
                top_gap,
                x="EV_per_station",
                y=city_col,
                orientation="h",
                color=prov_col if prov_col else None,
                labels={"EV_per_station": "EV per station"},
                height=600,
            )
            fig2.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)


# ---------- PAGE 3: Plug / Port Mix ---------- #

# ---------- PAGE 3: Plug / Port Mix ---------- #

elif page.startswith("3️⃣"):
    st.title("Plug & Port Mix")

    # Use plugs_detail if available; else use ports_detail
    df = None
    if plugs_df is not None:
        df = apply_province_filter(plugs_df).copy()
    elif ports_df is not None:
        df = apply_province_filter(ports_df).copy()

    if df is None:
        st.error("Neither `plugs_detail.csv` nor `ports_detail.csv` found.")
    else:
        # detect Level column
        plug_col = None
        for c in df.columns:
            if c.lower() == "level":
                plug_col = c
                break

        if plug_col is None:
            st.error("No 'Level' column found.")
            st.write(list(df.columns))
        else:
            st.write("Using **Level** as charging type classification")

            # FIXED: correct the count dataframe
            counts = df[plug_col].value_counts().reset_index()
            counts.columns = ["Level", "Count"]   # RENAME properly

            st.write("Counts DF:")
            st.dataframe(counts)

            st.subheader("Charging Level Distribution")

            # final working plot
            fig = px.bar(
                counts,
                x="Level",
                y="Count",
                color="Level",
                height=500,
                labels={"Level": "Charging Type Level", "Count": "Number of Chargers"}
            )
            st.plotly_chart(fig, use_container_width=True)





# ---------- PAGE 4 : Map of EV Stations ---------- #

elif page.startswith("4️⃣"):
    st.title("Map of EV Charging Stations")

    df = stations_df.copy()

    # Rename columns to match Streamlit requirements
    df = df.rename(columns={
        "Lat": "latitude",
        "Long": "longitude",
        "lat": "latitude",
        "long": "longitude"
    })

    # Check if the renamed columns exist
    if "latitude" not in df.columns or "longitude" not in df.columns:
        st.error("Latitude/Longitude columns not found even after renaming.")
        st.write("Columns:", list(df.columns))
    else:
        st.success("Using coordinates: latitude, longitude")

        # Convert to numeric
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

        # Drop rows with missing coordinates
        df = df.dropna(subset=["latitude", "longitude"])

        if df.empty:
            st.warning("No stations available to display after filtering.")
        else:
            st.subheader("Charging Stations Map")

            # Streamlit map now works 100%
            st.map(df[["latitude", "longitude"]])

            st.write("Plotted stations:", len(df))


# ---------- PAGE 5: Data Explorer ---------- #

elif page.startswith("5️⃣"):
    st.title("Raw Data Explorer")

    dataset_name = st.selectbox(
        "Choose a dataset",
        [
            "stations_detail.csv",
            "ports_detail.csv",
            "plugs_detail.csv",
            "ev_city_station_summary.csv",
            "canadacities.csv",
        ],
    )

    df_map = {
        "stations_detail.csv": stations_df,
        "ports_detail.csv": ports_df,
        "plugs_detail.csv": plugs_df,
        "ev_city_station_summary.csv": summary_df,
        "canadacities.csv": cities_df,
    }

    df = df_map[dataset_name]

    if df is None:
        st.error(f"`{dataset_name}` not found in {DATA_DIR}")
    else:
        df = apply_province_filter(df)

        st.write(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(100))

        st.subheader("Summary statistics")
        with st.expander("Show describe()"):
            st.write(df.describe(include="all"))

 # ---------- PAGE 6: Rural Demand Index (RDI) Model ---------- #

# ---------- PAGE 6: Rural Demand Index (RDI) Model ---------- #

# ---------- PAGE 6: Rural Demand Index (RDI) Model ---------- #

elif page.startswith("6️⃣"):
    st.title("📊 Rural Demand Index (RDI) Model")

    if summary_df is None or cities_df is None:
        st.error("Required datasets missing: ev_city_station_summary.csv and canadacities.csv")
    else:
        st.success("RDI model using EV vs stations + city demographics")

        # -----------------------
        # FIX COLUMN NAMES
        # -----------------------
        cities_df_clean = cities_df.rename(columns={
            "city": "City",
            "province_name": "Province",
            "population": "Population",
        })

        # -----------------------
        # NORMALIZE TEXT (critical!)
        # -----------------------
        summary_df["City"] = summary_df["City"].str.lower().str.strip()
        summary_df["Province"] = summary_df["Province"].str.lower().str.strip()

        cities_df_clean["City"] = cities_df_clean["City"].str.lower().str.strip()
        cities_df_clean["Province"] = cities_df_clean["Province"].str.lower().str.strip()

        st.write("City dataset columns:", list(cities_df_clean.columns))

        # -----------------------
        # MERGE PROPERLY
        # -----------------------
        merged = pd.merge(
            summary_df,
            cities_df_clean[["City", "Province", "Population"]],
            on=["City", "Province"],
            how="left"
        )

        st.subheader("Merged Dataset Preview")
        st.dataframe(merged.head())

        # -----------------------
        # HANDLE MISSING POPULATION
        # -----------------------
        merged["Population"] = pd.to_numeric(merged["Population"], errors="coerce")
        merged["Population"] = merged["Population"].fillna(merged["Population"].median())

        # -----------------------
        # FEATURE ENGINEERING
        # -----------------------
        st.subheader("🧮 Feature Engineering")

        merged["EV_per_capita"] = merged["EV_Count"] / merged["Population"]
        merged["Distance_Score"] = merged["Population"] / (merged["Charging_Stations"] + 1)
        merged["Accessibility"] = merged["Charging_Stations"] / (merged["EV_Count"] + 1)

        # Normalize features
        for col in ["EV_per_capita", "Distance_Score", "Accessibility"]:
            merged[f"{col}_norm"] = (
                merged[col] - merged[col].min()
            ) / (merged[col].max() - merged[col].min() + 1e-9)

        # Final RDI score
        merged["RDI"] = (
            0.40 * merged["EV_per_capita_norm"] +
            0.40 * merged["Distance_Score_norm"] +
            0.20 * merged["Accessibility_norm"]
        )

        st.write("Computed RDI Scores:")
        st.dataframe(merged[["City", "Province", "RDI"]].head())

        # -----------------------
        # RDI DISTRIBUTION
        # -----------------------
        st.subheader("📈 RDI Distribution")
        fig = px.histogram(merged, x="RDI", nbins=20, title="Distribution of RDI Scores")
        st.plotly_chart(fig, use_container_width=True)



        # -----------------------
        # TOP UNDERSERVED REGIONS
        # -----------------------
        st.subheader("🚨 Top Underserved Regions by RDI")

        unders = merged.sort_values("RDI", ascending=False).head(15)
        st.dataframe(unders[["City", "Province", "RDI"]])

        st.plotly_chart(px.bar(
            unders,
            x="City",
            y="RDI",
            color="Province",
            title="Top Underserved Cities"
        ))

        # -----------------------
        # DOWNLOAD BUTTON
        # -----------------------
        st.download_button(
            "📥 Download RDI Results",
            merged.to_csv(index=False),
            file_name="RDI_results.csv",
            mime="text/csv"
        )



