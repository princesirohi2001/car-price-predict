import pandas as pd
import pickle as pk
import streamlit as st
import time

# Streamlit Page Config
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Custom Styling & Fonts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron&display=swap');

    html, body {
        background-image: url('https://images.unsplash.com/photo-1517949908114-720e2c4e0f87');
        background-size: cover;
        background-attachment: fixed;
        font-family: 'Orbitron', sans-serif;
        color: white;
    }
    .main {
        background: rgba(0, 0, 0, 0.65);
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
    .stButton>button {
        background-color: #00ffcc;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px 24px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #00e6ac;
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 30px;
        font-size: 12px;
        color: #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Logo
st.sidebar.image("logo.png", use_container_width=True)

st.sidebar.markdown("### 🤖 Car Price AI\n##### By Prince Sirohi")

# Load model
try:
    model = pk.load(open("model.pkl", "rb"))
except FileNotFoundError:
    st.error("❌ Model file not found.")
    st.stop()

# Load data
try:
    df = pd.read_csv("cardekho.csv")
except FileNotFoundError:
    st.error("❌ Dataset file not found.")
    st.stop()

df['name'] = df['name'].apply(lambda x: x.split(' ')[0].strip())

# App Header
st.markdown("<h1 style='text-align:center;'>🚘 Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center; color:#ccc;'>Estimate car resale value smartly with AI + Insurance logic</div>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Form Container
with st.container():
    with st.form("car_input_form"):
        st.markdown("<div class='main'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            name = st.selectbox('🚗 Car Brand', sorted(df['name'].unique()))
            year = st.slider('📅 Year of Manufacture', 1994, 2024, 2017)
            km_driven = st.slider('🛣️ Kilometers Driven', 0, 200000, 50000)
            fuel = st.selectbox('⛽ Fuel Type', df['fuel'].unique())
            seller_type = st.selectbox('👤 Seller Type', df['seller_type'].unique())

        with col2:
            transmission = st.selectbox('⚙️ Transmission', df['transmission'].unique())
            owner = st.selectbox('👥 Owner Type', df['owner'].unique())
            mileage = st.slider('📊 Mileage (km/ltr/kg)', 10, 40, 18)
            engine = st.slider('🔧 Engine (cc)', 700, 5000, 1200)
            max_power = st.slider('💪 Max Power (BHP)', 100, 200, 120)
            seats = st.slider('🪑 Number of Seats', 2, 10, 5)
            insurance_claims = st.slider('📝 Insurance Claims', 0, 10, 0)

        st.markdown("</div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Predict Now")

# Prediction
if submitted:
    input_data_model = pd.DataFrame(
        [[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]],
        columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats']
    )

    # Encode categorical values
    input_data_model['name'].replace(
        ['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra', 'Tata', 'Chevrolet',
         'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
         'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
        list(range(1, 32)),
        inplace=True
    )
    input_data_model['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    input_data_model['owner'].replace(
        ['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'],
        [1, 2, 3, 4, 5],
        inplace=True
    )
    input_data_model['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    input_data_model['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)

    input_data_model.rename(columns={'mileage': 'mileage(km/ltr/kg)'}, inplace=True)

    # Spinner with loading gif
    with st.spinner("🔍 Analyzing car data..."):
        st.markdown("""
            <div style='text-align: center;'>
                <img src='https://i.gifer.com/Wn55.gif' width='120'>
                <p style='color:white;'>Running AI model...</p>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(2)

        try:
            car_price = model.predict(input_data_model)[0]
            adjusted_price = car_price * (1 - (0.02 * insurance_claims))

            st.success("✅ Prediction complete!")
            st.markdown("<h2 style='color:#00ffcc;'>💸 Estimated Car Price</h2>", unsafe_allow_html=True)
            st.markdown(f"💰 **Before insurance adjustment:** ₹{car_price:,.2f}")
            st.markdown(f"🧾 **After insurance claims adjustment:** ₹{adjusted_price:,.2f}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")

# Footer
st.markdown("<div class='footer'>🚀 Built by Prince Sirohi • 2025</div>", unsafe_allow_html=True)
