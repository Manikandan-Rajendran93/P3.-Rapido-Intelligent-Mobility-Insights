import streamlit as st
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy import text
import plotly.express as px
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

st.markdown(
    """
    <style>
    .block-container {
        padding-left: 5rem;
        padding-right: 5rem;
        padding-top: 5rem;
        padding-bottom: 5rem;
        max-width: 90%;
    }
    </style>
    """,
    unsafe_allow_html=True
)

engine = create_engine("mysql+mysqlconnector://root:9095@localhost/rapido")

st.set_page_config(
    page_title = "RAPIDO MOBILITY INSIGHTS",
    page_icon = "🚕"
)

def fetch_data(query, params = None):
    sql_text = text(query)
    df = pd.read_sql(sql_text, engine, params = params)
    return df

@st.cache_data
def get_data(query, params = None):
    return fetch_data(query, params)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)        
local_css("style.css")

st.markdown('<p class="main-header">🚕 RAPIDO MOBILITY INSIGHTS</p>', unsafe_allow_html=True)

[tab_fare, tab_driver, tab_ride_outcome, tab_customer, tab_visuals] = st.tabs(
    ["FARE PREDICTION", "DRIVER DELAY PREDICTION",
    "RIDE OUTCOME PREDICTION", "CUSTOMER CANCEL PREDICTION", "VISUAL INSIGHTS"]
    )

## FARE PREDICTION ##:
with tab_fare:
    X = get_data("""select 
        day_of_week, city, vehicle_type, ride_distance_km, 
        estimated_ride_time_min, traffic_level, weather_condition, 
        base_fare, surge_multiplier 
        from bookings""")

    y = get_data("select booking_value from bookings")

    # train test split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # PreProcessing Train data:
    # Encoding Categorical features
    categorical_features = ["day_of_week", "city", "vehicle_type", "traffic_level", "weather_condition"]
    OHE = OneHotEncoder(sparse_output = False)
    X_train_ohe_array = OHE.fit_transform(X_train[categorical_features])
    X_train_encoded = pd.DataFrame(
        X_train_ohe_array, 
        columns = OHE.get_feature_names_out(), 
        index = X_train.index
        )

    # Scaling Numerical features:
    numerical_features = ["ride_distance_km", "estimated_ride_time_min", "base_fare", "surge_multiplier"]
    MMS = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        MMS.fit_transform(X_train[numerical_features]), 
        columns = numerical_features, 
        index = X_train.index
        )

    # Concatinate encoded and scaled features together:
    X_train_final = pd.concat([X_train_scaled, X_train_encoded], axis = 1)

    # Preprocessing Test data:
    # Encoding Categorical features
    X_test_ohe_array = OHE.transform(X_test[categorical_features])
    X_test_encoded = pd.DataFrame(
        X_test_ohe_array, 
        columns = OHE.get_feature_names_out(), 
        index = X_test.index
        )

    # Scaling Numerical features:
    X_test_scaled = pd.DataFrame(
        MMS.transform(X_test[numerical_features]), 
        columns = numerical_features, 
        index = X_test.index
        )

    # Concatinate encoded and scaled features together:
    X_test_final = pd.concat([X_test_scaled, X_test_encoded], axis = 1)

    fare_prediction_model = joblib.load("fare_sgd.pkl")
    y_pred = pd.DataFrame(
        fare_prediction_model.predict(X_test_final), 
        columns = y_test.columns, 
        index = X_test_final.index
        )

    R2_score_test = fare_prediction_model.score(X_test_final, y_test)
    R2_score_train = fare_prediction_model.score(X_train_final, y_train)
    MAE = mean_absolute_error(y_pred, y_test)
    RMSE = root_mean_squared_error(y_pred, y_test)
    percentage_errors = np.abs(y_test - y_pred) / y_test
    MdAPE = np.median(percentage_errors) * 100


    st.markdown('<p class="custom-subheader">FARE PREDICTION MODEL</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">EVALUATION METRICS</p>', unsafe_allow_html=True)
    kpi_columns = st.columns(4)
    kpi_columns[0].metric(label = "R2 SCORE ON TEST DATA", value = f'{round(R2_score_test, 2)}')
    kpi_columns[1].metric(label = "R2 SCORE ON TRAIN DATA", value = f'{round(R2_score_train, 2)}')
    kpi_columns[2].metric(label = "MAE", value = f'{round(MAE, 2)} INR')
    kpi_columns[3].metric(label = "RMSE", value = f'{round(RMSE, 2)} INR')

    st.markdown('<p class="custom-text">BUSINESS INSIGHT</p>', unsafe_allow_html=True)
    kpi_columns = st.columns(1)
    kpi_columns[0].metric(label = "MEDIAN ABSOLUTE PERCENTAGE ERROR", value = f'{round(MdAPE, 2)} %')

## DRIVER DELAY PREDICTION ##:
with tab_driver:
    df = get_data("""select 
    booking_id,	booking_timestamp, day_of_week,
    is_weekend,	hour_of_day, city, pickup_location,	
    drop_location,vehicle_type, ride_distance_km, 
    estimated_ride_time_min, traffic_level, weather_condition, 
    base_fare,surge_multiplier, booking_value, 
    booking_status, incomplete_ride_reason, customer_id, 
    driver_id, total_assigned_rides, accepted_rides, 
    driver_incomplete_rides,driver_delay_count,
    driver_acceptance_rate,driver_delay_rate,
    avg_driver_rating, avg_pickup_delay_min,
    driver_delay_flag
    from combined_features""")

    df["risk_of_delay"] = (df["driver_delay_rate"] > 0.15).astype(int)

    X = df.drop(columns = ["risk_of_delay"])
    y = df[["risk_of_delay"]]
    
    # traintest split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    # preprocess training data     
    # categorical encoding
    categorical_features = ["day_of_week", "city", "vehicle_type", "traffic_level", "weather_condition"]

    OHEX = OneHotEncoder(sparse_output = False)
    X_train_ohe_array = OHEX.fit_transform(X_train[categorical_features])
    X_train_encoded = pd.DataFrame(
        X_train_ohe_array, 
        columns = OHEX.get_feature_names_out(), 
        index = X_train.index)
    
    # numerical scaling
    numerical_features = ["hour_of_day", "ride_distance_km", "estimated_ride_time_min", 
        "base_fare", "surge_multiplier", "booking_value", 
        "total_assigned_rides", "accepted_rides", "driver_incomplete_rides", 
        "driver_delay_count", "driver_acceptance_rate", "driver_delay_rate",
        "avg_driver_rating", "avg_pickup_delay_min",
        "driver_delay_flag"]
    
    MMS = MinMaxScaler()
    X_train_scaled = pd.DataFrame(
        MMS.fit_transform(X_train[numerical_features]), 
        columns = numerical_features, 
        index = X_train.index)
    
    # concatinate
    X_train_final = pd.concat([X_train_scaled, X_train_encoded], axis = 1)

    # preprocess test data
    # categorical encoding
    X_test_ohe_array = OHEX.transform(X_test[categorical_features])
    X_test_encoded = pd.DataFrame(
        X_test_ohe_array, 
        columns = OHEX.get_feature_names_out(), 
        index = X_test.index)

    # numerical scaling
    X_test_scaled = pd.DataFrame(
        MMS.transform(X_test[numerical_features]), 
        columns = numerical_features, 
        index = X_test.index)

    ## concatinate
    X_test_final = pd.concat([X_test_scaled, X_test_encoded], axis = 1)

    features = ["driver_delay_rate", "avg_driver_rating","avg_pickup_delay_min", 
     "driver_acceptance_rate", "hour_of_day", "ride_distance_km",
     "surge_multiplier", "traffic_level_High", "estimated_ride_time_min",
     "traffic_level_Low", "traffic_level_Medium"]

    driver_delay_prediction_model = joblib.load("driver_delay_logreg.pkl")
    y_pred = pd.DataFrame(
        driver_delay_prediction_model.predict(X_test_final[features]), 
        columns = y_train.columns, index = X_test_final.index
        )
    F1_score_train = driver_delay_prediction_model.score(
        X_train_final[features], y_train.values.ravel()
        )
    F1_score_test = driver_delay_prediction_model.score(
        X_test_final[features], y_test.values.ravel()
        )
    
    st.markdown('<p class="custom-subheader">DRIVER DELAY PREDICTION MODEL</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">EVALUATION METRICS</p>', unsafe_allow_html=True)
    kpi_columns = st.columns(2)
    kpi_columns[0].metric(label = "F1 SCORE ON TRAIN DATA", value = f'{round(F1_score_train, 3)}')
    kpi_columns[1].metric(label = "F1 SCORE ON TEST DATA", value = f'{round(F1_score_test, 3)}')
    
    classification_dict = classification_report(
        y_test, y_pred, target_names ={"no risk of delay": 0, "risk of delay": 1}, output_dict = True)
    classification_report_driver = pd.DataFrame(classification_dict).T
        
    st.markdown('<p class="custom-text">CLASSIFICATION REPORT</p>', unsafe_allow_html=True)
    st.dataframe(classification_report_driver)

    st.markdown('<p class="custom-text">CONFUSION MATRIX</p>', unsafe_allow_html=True)
    cm = confusion_matrix(y_pred, y_test)
    st.dataframe(pd.DataFrame(
        cm, 
        columns = ["Predicted: No Risk", "Predicted: Risk"], 
        index = ["Actual: No Risk", "Actual: Risk"]
        ))

## CUSTOMER CANCEL PREDICTION ##    
with tab_customer:
    df = get_data("""select 
        booking_id,	booking_timestamp, day_of_week,
        is_weekend,	hour_of_day, city, pickup_location,	drop_location,
        vehicle_type, ride_distance_km,	estimated_ride_time_min,
        traffic_level, weather_condition, base_fare,
        surge_multiplier, booking_value, booking_status,
        incomplete_ride_reason,	customer_id, driver_id,
        customer_total_bookings, customer_completed_rides,
        customer_cancelled_rides, customer_incomplete_rides,
        customer_cancellation_rate, avg_customer_rating,
        customer_cancel_flag
        from combined_features""")

    # Create a true binary target: 1 if Cancelled, 0 for anything else (Completed/Incomplete)
    df["booking_status"] = (df['booking_status'] == 'Cancelled').astype(int)
    y = df[["booking_status"]]
    X = df.drop(columns = ["booking_status"])
    
    # train test splt
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    
    # preprocessing train data
    # categorical encoding - feature
    categorical_features = ["day_of_week", "city", "vehicle_type", "traffic_level", "weather_condition"]

    OHEX = OneHotEncoder(sparse_output = False)
    X_train_ohe_array = OHEX.fit_transform(X_train[categorical_features])
    X_train_encoded = pd.DataFrame(
        X_train_ohe_array, 
        columns = OHEX.get_feature_names_out(), 
        index = X_train.index)
    
    # numerical scaling - features
    numerical_features = ["hour_of_day", "ride_distance_km", "estimated_ride_time_min", 
            "base_fare", "surge_multiplier", "booking_value", 
            "customer_total_bookings", "customer_completed_rides", "customer_cancelled_rides", 
            "customer_incomplete_rides", "customer_cancellation_rate", "avg_customer_rating", 
            "customer_cancel_flag"
            ]
    MMS = MinMaxScaler()
    X_train_scaled = pd.DataFrame(MMS.fit_transform(
        X_train[numerical_features]), 
        columns = numerical_features, 
        index = X_train.index)
    
    # Concatinating encoded and scaled features together:
    X_train_final = pd.concat([X_train_scaled, X_train_encoded], axis = 1)
    # y_train - Target label - already numerical

    # Preprocessing Test data:
    # categorical encoding - featutres
    X_test_ohe_array = OHEX.transform(X_test[categorical_features])
    X_test_encoded = pd.DataFrame(
        X_test_ohe_array, 
        columns = OHEX.get_feature_names_out(), 
        index = X_test.index)
    
    # numerical scaling - features
    X_test_scaled = pd.DataFrame(MMS.transform(
        X_test[numerical_features]), 
        columns = numerical_features, 
        index = X_test.index)
    
    # Concatinate encoded and scaled features together:
    X_test_final = pd.concat([X_test_scaled, X_test_encoded], axis = 1)
    # y_test - Target label - already numerical

    features = ["customer_cancellation_rate", "surge_multiplier", 
        "customer_cancelled_rides", "customer_completed_rides", 
        "customer_cancel_flag"]
    
    customer_cancel_prediction_model = joblib.load("customer_cancellation_logreg.pkl")
    y_pred = pd.DataFrame(
        customer_cancel_prediction_model.predict(X_test_final[features]), 
        columns = y_pred.columns, index = X_test_final.index)
    F1_score_train = customer_cancel_prediction_model.score(
        X_train_final[features], y_train.values.ravel()
        )
    F1_score_test = customer_cancel_prediction_model.score(
        X_test_final[features], y_test.values.ravel()
        )
    
    st.markdown('<p class="custom-subheader">CUSTOMER CANCEL PREDICTION MODEL</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">EVALUATION METRICS</p>', unsafe_allow_html=True)
    kpi_columns = st.columns(2)
    kpi_columns[0].metric(label = "F1 SCORE ON TRAIN DATA", value = f'{round(F1_score_train, 2)}')
    kpi_columns[1].metric(label = "F1 SCORE ON TEST DATA", value = f'{round(F1_score_test, 2)}')
    
    classification_dict = classification_report(
        y_test, y_pred, target_names ={"no risk of cancel": 0, "risk of cancel": 1}, output_dict = True)
    classification_report_customer = pd.DataFrame(classification_dict).T
    
    st.markdown('<p class="custom-text">CLASSIFICATION REPORT</p>', unsafe_allow_html=True)
    st.dataframe(classification_report_customer)
    
    st.markdown('<p class="custom-text">CONFUSION MATRIX</p>', unsafe_allow_html=True)
    cm = confusion_matrix(y_pred, y_test)
    st.dataframe(pd.DataFrame(
        cm, 
        columns = ["Predicted: No Risk", "Predicted: Risk"], 
        index = ["Actual: No Risk", "Actual: Risk"]
        ))

## RIDE OUTCOME PREDICTION ##    
with tab_ride_outcome:
    ## import data from sql
    df = get_data("select * from combined_features")
    X = df.drop(columns = ["booking_status"])

    y = get_data("select booking_status from bookings")

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
    X_train = X_train.drop(columns = ["actual_ride_time_min"])
    
    # preprocessing train data
    # categorical encoding - feature matrix
    categorical_features = ["day_of_week", "city", "vehicle_type", "traffic_level", "weather_condition"]

    OHEX = OneHotEncoder(sparse_output = False)
    X_train_ohe_array = OHEX.fit_transform(X_train[categorical_features])
    X_train_encoded = pd.DataFrame(
        X_train_ohe_array, 
        columns = OHEX.get_feature_names_out(), 
        index = X_train.index)
    
    # numerical scaling - feature matrix
    numerical_features = ["hour_of_day", "ride_distance_km", "estimated_ride_time_min", 
            "base_fare", "surge_multiplier", "booking_value", 
            "customer_total_bookings", "customer_completed_rides", "customer_cancelled_rides", 
            "customer_incomplete_rides", "customer_cancellation_rate", "avg_customer_rating", 
            "total_assigned_rides", "accepted_rides", "driver_incomplete_rides", 
            "driver_delay_count", "driver_acceptance_rate", "driver_delay_rate",
            "avg_driver_rating", "avg_pickup_delay_min", "customer_cancel_flag",
            "driver_delay_flag"]
    MMS = MinMaxScaler()
    X_train_scaled = pd.DataFrame(MMS.fit_transform(
        X_train[numerical_features]), 
        columns = numerical_features, 
        index = X_train.index)
    
    # Concatinating encoded and scaled features together:
    X_train_final = pd.concat([X_train_scaled, X_train_encoded], axis = 1)
    
    # Categorical encoding - Target label
    encoder = LabelEncoder()
    y_train_encoded_array = encoder.fit_transform(y_train.values.ravel())
    y_train_encoded = pd.DataFrame(
        y_train_encoded_array, 
        columns = y_train.columns, 
        index = y_train.index)
    
    # Preprocessing Test data:
    # categorical encoding - featutre matrix
    X_test_ohe_array = OHEX.transform(X_test[categorical_features])
    X_test_encoded = pd.DataFrame(
        X_test_ohe_array, 
        columns = OHEX.get_feature_names_out(), 
        index = X_test.index)
    
    # numerical scaling - feature matrix
    X_test_scaled = pd.DataFrame(MMS.transform(
        X_test[numerical_features]), 
        columns = numerical_features, 
        index = X_test.index)
    
    # Concatinate encoded and scaled features together:
    X_test_final = pd.concat([X_test_scaled, X_test_encoded], axis = 1)
    
    # Preprocessing Test data:
    # Categorical encoding - Target label
    y_test_encoded_array = encoder.transform(y_test.values.ravel())
    y_test_encoded = pd.DataFrame(
        y_test_encoded_array, 
        columns = y_test.columns, 
        index = y_test.index)
    
    features = ["customer_cancellation_rate", "surge_multiplier", "customer_incomplete_rides",
        "customer_cancelled_rides", "driver_acceptance_rate",  
        "driver_incomplete_rides", "customer_completed_rides", "driver_delay_rate",
        "driver_delay_count", "traffic_level_High", "traffic_level_Low", 
        "traffic_level_Medium"]
    
    ride_outcome_prediction_model = joblib.load("ride_outcome_logreg.pkl")
    y_pred = pd.DataFrame(
        ride_outcome_prediction_model.predict(X_test_final[features]), 
        columns = y_train.columns, 
        index = X_test_final.index)
    
    F1_score_train = ride_outcome_prediction_model.score(
        X_train_final[features], y_train_encoded.values.ravel()
        )
    F1_score_test = ride_outcome_prediction_model.score(
        X_test_final[features], y_test_encoded.values.ravel()
        )
    
    st.markdown('<p class="custom-subheader">RIDE OUTCOME PREDICTION MODEL</p>', unsafe_allow_html=True)
    st.markdown('<p class="custom-text">EVALUATION METRICS</p>', unsafe_allow_html=True)
    kpi_columns = st.columns(2)
    kpi_columns[0].metric(label = "F1 SCORE ON TRAIN DATA", value = f'{round(F1_score_train, 2)}')
    kpi_columns[1].metric(label = "F1 SCORE ON TEST DATA", value = f'{round(F1_score_test, 2)}')
    
    classification_dict = classification_report(
        y_test_encoded, y_pred, target_names = encoder.classes_, output_dict = True)
    classification_report_ride_outcome = pd.DataFrame(classification_dict).T
        
    st.markdown('<p class="custom-text">CLASSIFICATION REPORT</p>', unsafe_allow_html=True)
    st.dataframe(classification_report_ride_outcome)

    st.markdown('<p class="custom-text">CONFUSION MATRIX</p>', unsafe_allow_html=True)
    cm = confusion_matrix(y_pred, y_test_encoded)
    st.dataframe(pd.DataFrame(
        cm, 
        columns = ["Predicted: Cancelled", "Predicted: Completed", "Predicted: Incomplete"], 
        index = ["Actual: Cancelled", "Actual: Completed", "Actual: Incomplete"]
        ))

# VISULAS
with tab_visuals:
    chart_column1, chart_column2, chart_column3 = st.columns(3)  
    with chart_column1:
        ride_day_df = get_data("""select
        day_of_week, 
        count(*) as no_of_rides
        from bookings 
        where booking_status = 'Completed' 
        group by day_of_week"""
        )
        st.markdown('<p class="custom-text">RIDE VOLUME BY WEEKDAY</p>', unsafe_allow_html=True)
        fig_ride_day = px.bar(
            ride_day_df, x = "day_of_week", y = "no_of_rides",
            labels = {"day_of_week": "DAYS", "no_of_rides": "RIDE VOLUME"}
            )
        st.plotly_chart(fig_ride_day)

    with chart_column2:
        ride_city_df = get_data("""select
        city,
        count(*) as no_of_rides
        from bookings
        where booking_status = 'Completed'
        group by city"""
        )
        st.markdown('<p class="custom-text">RIDE VOLUME BY CITY</p>', unsafe_allow_html=True)
        fig_ride_city = px.bar(
            ride_city_df, x = "city", y = "no_of_rides",
            labels = {"city": "CITY", "no_of_rides": "RIDE VOLUME"}
            )
        fig_ride_city.update_xaxes(tickangle = 45)
        st.plotly_chart(fig_ride_city)

    with chart_column3:
        ride_hour_df = get_data("""select
        date_format(booking_timestamp, "%H:00") as hour_block,
        count(*) as no_of_rides
        from bookings
        group by hour_block
        order by hour_block asc"""
        )
        st.markdown('<p class="custom-text">RIDE VOLUME BY HOUR</p>', unsafe_allow_html=True)
        fig_ride_hour = px.line(
            ride_hour_df, x = "hour_block", y = "no_of_rides",
            labels = {"hour_block": "TIMEZONE", "no_of_rides": "RIDE VOLUME"},
            markers = True)
        fig_ride_hour.update_xaxes(tickangle = 35)
        st.plotly_chart(fig_ride_hour)

    st.divider()

    chart_column1, chart_column2, chart_column3 = st.columns(3)  
        
    with chart_column1:
        cancel_traffic_df = get_data("""select
        traffic_level, 
        count(*) as no_of_cancellations
        from bookings
        where booking_status = 'Cancelled'
        group by traffic_level"""
        )
        st.markdown('<p class="custom-text">CANCELLATIONS BY TRAFFIC LEVEL</p>', unsafe_allow_html=True)
        fig_cancel_traffic = px.pie(
            cancel_traffic_df, names = "traffic_level", values = "no_of_cancellations"
            )
        fig_cancel_traffic.update_traces(
            textposition = "inside", textinfo = "percent+label"
            )
        st.plotly_chart(fig_cancel_traffic)

    with chart_column2:
        cancel_city_df = get_data("""select 
        city,
        count(*) as no_of_cancellations
        from bookings
        where booking_status = 'Cancelled'
        group by city"""
        )
        st.markdown('<p class="custom-text">CANCELLATIONS BY CITY</p>', unsafe_allow_html=True)
        fig_cancel_city = px.pie(
            cancel_city_df, names = "city", values = "no_of_cancellations"
            )
        fig_cancel_city.update_traces(
            textposition = "inside", textinfo = "percent+label"
            )
        st.plotly_chart(fig_cancel_city)

    with chart_column3:
        cancel_weather_df = get_data("""select 
        weather_condition,
        count(*) as no_of_cancellations
        from bookings
        where booking_status = 'Cancelled'
        group by weather_condition"""
        )
        st.markdown('<p class="custom-text">CANCELLATIONS BY WEATHER</p>', unsafe_allow_html=True)
        fig_cancel_weather = px.pie(
            cancel_weather_df, names = "weather_condition", values = "no_of_cancellations"
            )
        fig_cancel_weather.update_traces(
            textposition = "inside", textinfo = "percent+label"
            )
        st.plotly_chart(fig_cancel_weather)

    st.divider()

    coulumn_chart1 = st.columns(1)

    with chart_column1:
        fare_distance_df = get_data("""select
        booking_value, 
        ride_distance_km
        from bookings"""
        )
        st.markdown('<p class="custom-text">FARE VS DISTANCE</p>', unsafe_allow_html=True)
        fig_fare_distance = px.line(
            fare_distance_df, x = "ride_distance_km", y = "booking_value",
            labels = {"ride_distance_km": "DISTANCE KM", "booking_value": "FARE RUPEES"},
            markers = True)
        st.plotly_chart(fig_fare_distance, use_container_width = True)

    