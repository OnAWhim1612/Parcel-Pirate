import streamlit as st
import pandas as pd
import googlemaps
from itertools import combinations
import random
import numpy as np
from geopy.distance import great_circle

# Load the CSV files into DataFrames
fleet_df = pd.read_csv("fleet.csv")
deliveries_df = pd.read_csv("deliveries.csv")
warehouses_df = pd.read_csv("warehouses.csv")

# Initialize the Google Maps client with your API key
gmaps = googlemaps.Client(key='AIzaSyADQUIDM3-oA6tQcjA4T_ZQ-Uvw_XPZg98')

carbon_emission_rate = 0.5
def calculate_emissions(route_length_km, carbon_emission_rate):
    return route_length_km * carbon_emission_rate
# Calculate Shortest Route
def calculate_shortest_routes(warehouses_df, deliveries_df):
    routes = []

    warehouse_locations = warehouses_df['Location'].tolist()
    warehouse_names = warehouses_df['Warehouse'].tolist()
    destination_locations = deliveries_df['Des_Lat_Lon'].tolist()
    destination_names = deliveries_df['Destination_Location'].tolist()

    for warehouse, warehouse_name in zip(warehouse_locations, warehouse_names):
        for destination, destination_name in zip(destination_locations, destination_names):
            warehouse_coords = tuple(map(float, warehouse.split(', ')))
            destination_coords = tuple(map(float, destination.split(', ')))

            # Calculate the distance between warehouse and destination using great-circle distance
            distance_km = great_circle(warehouse_coords, destination_coords).kilometers

            routes.append({
                'Warehouse': warehouse_name,
                'Warehouse Location': warehouse,
                'Destination': destination_name,
                'Distance (km)': distance_km
            })

    return pd.DataFrame(routes)

# Calculate the number of parcels from each warehouse
parcels_per_warehouse = deliveries_df['Warehouse'].value_counts().reset_index()
parcels_per_warehouse.columns = ['Warehouse', 'Parcels']

# Determine the maximum vehicle capacity
max_vehicle_capacity = 3  # Maximum capacity of each vehicle

# Initialize a list to store the results for each warehouse
results = []

# Calculate the number of vehicles required for each warehouse
for index, row in parcels_per_warehouse.iterrows():
    parcels = row['Parcels']
    vehicles_required = parcels // max_vehicle_capacity
    remaining_parcels = parcels % max_vehicle_capacity

    # If there are remaining parcels, adjust the number of vehicles required
    if remaining_parcels > 0:
        vehicles_required += 1

    results.append({
        'Warehouse': row['Warehouse'],
        'Vehicles Required': vehicles_required
    })

# Create a DataFrame from the results
vehicles_required_df = pd.DataFrame(results)

# Function to calculate emissions
def calculate_emissions(route_length_km, carbon_emission_rate):
    return route_length_km * carbon_emission_rate

# Function to assign parcels to a single vehicle and calculate emissions
def assign_single_vehicle_delivery_streamlit(warehouse_name, warehouse_location, delivery_locations, assigned_driver, carbon_emission_rate):
    st.write(f"Assigning parcels for {warehouse_name} by Driver: {assigned_driver}:")
    st.write(f"Starting from Warehouse at {warehouse_location}")

    route_distance_km = 0  # Initialize route distance
    emissions = 0  # Initialize emissions

    while delivery_locations:
        closest_location = None
        min_distance = float('inf')

        # Find the closest location to the current location
        for location in delivery_locations:
            distance_km = shortest_routes_df[
                (shortest_routes_df['Warehouse'] == warehouse_name) &
                (shortest_routes_df['Destination'] == location)
                ]['Distance (km)'].values[0]

            if distance_km < min_distance:
                closest_location = location
                min_distance = distance_km

        st.write(f"Deliver to {closest_location}")
        delivery_locations.remove(closest_location)

        # Update route distance and emissions
        route_distance_km += min_distance
        emissions += calculate_emissions(min_distance, carbon_emission_rate)

    st.write(f"Total Route Distance: {route_distance_km} km")
    st.write(f"Total Emissions: {emissions} kg CO2")

# Function to assign parcels to multiple vehicles and calculate emissions
def assign_multiple_vehicle_delivery_streamlit(warehouse_name, warehouse_location, delivery_locations, assigned_drivers, carbon_emission_rate):
    st.write(f"Assigning parcels for {warehouse_name}:")

    # Sort delivery locations by distance to the warehouse
    delivery_locations = sorted(delivery_locations, key=lambda loc: shortest_routes_df[
        (shortest_routes_df['Warehouse'] == warehouse_name) &
        (shortest_routes_df['Destination'] == loc)
        ]['Distance (km)'].values[0])

    # Divide delivery locations among the assigned drivers
    num_drivers = len(assigned_drivers)
    parcels_per_driver = np.floor(len(delivery_locations) / num_drivers).astype(int)
    remaining_parcels = len(delivery_locations) % num_drivers

    for i, driver in enumerate(assigned_drivers):
        # Determine the parcels for this driver
        if i < remaining_parcels:
            driver_parcels = delivery_locations[i * (parcels_per_driver + 1):(i + 1) * (parcels_per_driver + 1)]
        else:
            driver_parcels = delivery_locations[i * parcels_per_driver:(i + 1) * parcels_per_driver]

        st.write(f"Driver: {driver} - Starting from Warehouse at {warehouse_location}")

        route_distance_km = 0  # Initialize route distance
        emissions = 0  # Initialize emissions

        for parcel_location in driver_parcels:
            st.write(f"Deliver to {parcel_location} by Driver: {driver}")

            # Calculate distance to the next location
            distance_km = shortest_routes_df[
                (shortest_routes_df['Warehouse'] == warehouse_name) &
                (shortest_routes_df['Destination'] == parcel_location)
                ]['Distance (km)'].values[0]

            # Update route distance and emissions
            route_distance_km += distance_km
            emissions += calculate_emissions(distance_km, carbon_emission_rate)

        st.write(f"Total Route Distance: {route_distance_km} km")
        st.write(f"Total Emissions: {emissions} kg CO2")

# Calculate Shortest Routes with destination names and warehouse names
shortest_routes_df = calculate_shortest_routes(warehouses_df, deliveries_df)

# Create a Streamlit app
st.title("Last-Mile Delivery Optimization")
st.subheader("Promoting Sustainability in Transportation")

# Create sections for different features (you can add more later)
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a Page", ["Home", "Route Optimization and Emission Tracking"])

# Define the content for each page
if page == "Home":
    st.write(
        "Welcome to the Last-Mile Delivery Optimization web app. Use the navigation on the left to explore different features.")

    # Add information about last-mile delivery
    st.subheader("Last-Mile Delivery")
    st.write(
        "The last-mile delivery is the final step in the delivery process, where products are transported from a distribution center or warehouse to the customer's doorstep.")

    # Add information about sustainability
    st.subheader("Sustainability")
    st.write(
        "We are committed to promoting sustainability in transportation. Our optimization algorithms help reduce carbon emissions by finding the most efficient delivery routes.")
elif page == "Route Optimization and Emission Tracking":
    st.subheader("Closest location from each warehouse")

    # Find the closest location for each warehouse
    closest_locations = shortest_routes_df.groupby('Warehouse')['Distance (km)'].idxmin()
    closest_locations_df = shortest_routes_df.loc[closest_locations]

    # Display the closest locations including warehouse names
    st.dataframe(closest_locations_df[['Warehouse', 'Warehouse Location', 'Destination', 'Distance (km)']])

    # Create a table with all warehouses, locations, and distances
    st.subheader("All Warehouses, Locations, and Distances")
    st.dataframe(shortest_routes_df[['Warehouse', 'Warehouse Location', 'Destination', 'Distance (km)']])

    # Calculate parcels per warehouse and display vehicles required
    st.subheader("Parcels Per Warehouse")
    st.dataframe(parcels_per_warehouse[['Warehouse', 'Parcels']])

    st.subheader("Vehicles Required Per Warehouse")
    st.dataframe(vehicles_required_df)

    route_info_dict = {}

    for warehouse_row in parcels_per_warehouse.iterrows():
        warehouse_name = warehouse_row[1]['Warehouse']
        warehouse_location = warehouses_df.loc[warehouses_df['Warehouse'] == warehouse_name, 'Location'].iloc[
            0]  # Fetch warehouse location

        parcels = deliveries_df[deliveries_df['Warehouse'] == warehouse_name]
        delivery_locations = list(parcels['Destination_Location'])

        if vehicles_required_df.loc[vehicles_required_df['Warehouse'] == warehouse_name, 'Vehicles Required'].iloc[
            0] == 1:
            assigned_driver = random.choice(fleet_df['Driver_Name'])
            assign_single_vehicle_delivery_streamlit(warehouse_name, warehouse_location, delivery_locations,
                                                     assigned_driver, carbon_emission_rate)
        else:
            assigned_drivers = random.sample(list(fleet_df['Driver_Name']), vehicles_required_df.loc[
                vehicles_required_df['Warehouse'] == warehouse_name, 'Vehicles Required'].iloc[0])
            assign_multiple_vehicle_delivery_streamlit(warehouse_name, warehouse_location, delivery_locations,
                                                       assigned_drivers, carbon_emission_rate)

            # Calculate the total length of each driver's route
            for driver in assigned_drivers:
                total_route_distance_km = 0

                for parcel_location in delivery_locations:
                    distance_km = shortest_routes_df[
                        (shortest_routes_df['Warehouse'] == warehouse_name) &
                        (shortest_routes_df['Destination'] == parcel_location)
                        ]['Distance (km)'].values[0]

                    total_route_distance_km += distance_km

                # Store the route information in the dictionary
                route_info_dict[driver] = {
                    'Warehouse': warehouse_name,
                    'Total Route Distance (km)': total_route_distance_km,
                }



st.sidebar.markdown("---")
st.sidebar.text("© 2023 Parcel Pirate")

# Run the app
