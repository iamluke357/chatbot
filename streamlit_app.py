import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import openai
import os
import matplotlib.pyplot as plt

# 🔐 Set your API key securely
openai.api_key = st.secrets.get("openai_api_key", "sk-REPLACE_WITH_YOUR_KEY")

DATA_FILE = "sales_data.csv"

# Load and save functions
def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=["price", "quantity", "production_cost"])

# Calculate discount suggestions based on past wait time behavior
def calculate_wait_time_discounts(df):
    if "wait_time_minutes" not in df.columns or "returned" not in df.columns:
        return pd.DataFrame()

    bins = np.arange(0, 65, 5)
    df['wait_bin'] = pd.cut(df['wait_time_minutes'], bins=bins)

    return_rates = df.groupby('wait_bin')['returned'].mean() * 100
    discounts = (100 - return_rates) / 100 * 30

    discount_table = pd.DataFrame({
        "Wait Time Range (min)": return_rates.index.astype(str),
        "Return Rate (%)": return_rates.values.round(2),
        "Suggested Discount (%)": discounts.values.round(2)
    })

    return discount_table

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

# Optimization logic
def optimize_price(df, chef_available=True, degree=2, grid_steps=1000):
    prices = df["price"].values
    quantities = df["quantity"].values
    production_cost = df["production_cost"].iloc[-1]

    # 1D grid for plotting and calculation
    P_grid = np.linspace(min(prices), max(prices), grid_steps)
    P_grid_reshaped = P_grid.reshape(-1, 1)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(prices.reshape(-1, 1))
    model = LinearRegression().fit(X_poly, quantities)
    Q_pred = model.predict(poly.transform(P_grid_reshaped))
    Q_pred = Q_pred if chef_available else Q_pred * 0.5

    gross_revenue = ((P_grid - production_cost) * Q_pred).clip(min=0)

    waiter_cost = 100
    chef_cost = 200 if chef_available else 100
    waiters_needed = np.ceil(Q_pred / 20)
    chefs_needed = np.ceil(Q_pred / 100)
    labor_cost = waiters_needed * waiter_cost + chefs_needed * chef_cost
    tips = gross_revenue * 0.1
    net_revenue = gross_revenue - labor_cost + tips

    idx = np.argmax(net_revenue)
    base_price = P_grid[idx]
    base_quantity = Q_pred[idx]
    base_profit = net_revenue[idx]

    # Apply average discount if wait time data is available
    avg_discount_pct = 0
    if os.path.exists("wait_time_data.csv"):
        wait_data = pd.read_csv("wait_time_data.csv")
        if "wait_time_minutes" in wait_data.columns and "returned" in wait_data.columns:
            wait_data['wait_bin'] = pd.cut(wait_data['wait_time_minutes'], np.arange(0, 65, 5))
            return_rates = wait_data.groupby('wait_bin')['returned'].mean() * 100
            avg_discount_pct = ((100 - return_rates) / 100 * 30).mean()

    discount = avg_discount_pct / 100
    adjusted_net = base_profit - (base_price * discount * base_quantity)

    return {
        "optimal_price": round(base_price, 2),
        "estimated_quantity": round(base_quantity, 1),
        "net_profit": round(adjusted_net, 2),
        "avg_discount_pct": round(avg_discount_pct, 2),
        "P_grid": P_grid,
        "net_revenue": net_revenue,
        "Q_pred": Q_pred
    }

# GPT summary
def explain_result_with_gpt(price, quantity, profit, chef_sick):
    system_prompt = "You are a pricing assistant for a restaurant."
    user_prompt = (
        f"The AI recommends a price of ${price:.2f}, which should sell around {quantity:.0f} dishes, "
        f"earning ${profit:.2f} in profit. {'The chef is out sick.' if chef_sick else 'The chef is working.'} "
        f"Summarize this insight for the manager in clear terms."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.set_page_config(page_title="Daily Price Optimizer", layout="centered")
st.title("📅 Daily Restaurant Price Tracker")

data = load_data()

# Daily manual input
st.subheader("📥 Enter Daily Data")
with st.form("daily_input_form"):
    price = st.number_input("Dish Price ($)", min_value=1.0, step=0.5)
    quantity = st.number_input("Dishes Sold", min_value=1, step=1)
    production_cost = st.number_input("Production Cost ($)", min_value=0.0, step=0.5)
    submitted = st.form_submit_button("Add Entry")

    if submitted:
        new_row = pd.DataFrame([[price, quantity, production_cost]], columns=["price", "quantity", "production_cost"])
        data = pd.concat([data, new_row], ignore_index=True)
        save_data(data)
        st.success("✅ Entry added!")

# Show all data
st.subheader("📊 Sales Data So Far")
st.dataframe(data)

# Optimization
if not data.empty:
    st.subheader("🔧Analytics")
    chef_available = st.checkbox("Is the main chef available today?", value=True)
    st.subheader("Polynomial Degree")
    degree = st.slider("",1, 4, 2)

    result = optimize_price(data, chef_available=chef_available, degree=degree)

    st.subheader("💰 Recommended Pricing")
    st.metric("Optimal Price", f"${result['optimal_price']:.2f}")
    st.metric("Expected Quantity", f"{result['estimated_quantity']:.0f} dishes")
    st.metric("Expected Net Profit", f"${result['net_profit']:.2f}")
    st.metric("Applied Discount", f"{result['avg_discount_pct']:.1f}%")

    # Plot
    fig, ax = plt.subplots()
    ax.plot(result["P_grid"], result["net_revenue"], label="Net Profit", color="green")
    ax.plot(result["P_grid"], result["Q_pred"], label="Quantity", color="orange", linestyle="--")
    ax.axvline(result["optimal_price"], color="red", linestyle="--", label="Optimal Price")
    ax.set_xlabel("Price")
    ax.set_title("Net Profit vs Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

#Load external wait time data
if os.path.exists("wait_time_data.csv"):
    st.subheader("🧾 Suggested Discounts Based on Wait Time")
    wait_data = pd.read_csv("wait_time_data.csv")
    discount_table = calculate_wait_time_discounts(wait_data)
    if not discount_table.empty:
        st.dataframe(discount_table)
    else:
        st.info("No valid wait time data found.")

# Clear data
if st.button("❌ Clear All Data"):
    save_data(pd.DataFrame(columns=["price", "quantity", "production_cost"]))
    st.success("All data cleared.")
    st.experimental_rerun()
