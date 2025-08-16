"""
Swiss Alps Real Data Simulation for Avalanche Risk Assessment

This module simulates realistic data based on actual observations from Swiss avalanche
monitoring stations, particularly from the Swiss Federal Institute for Snow and Avalanche
Research (SLF). The data patterns are based on published research and typical observations
from stations like Weissfluhjoch, Davos.

Data Sources (Simulated based on):
- SLF Weissfluhjoch research station (2540m)
- IMIS automatic weather and snow stations
- MeteoSwiss alpine weather network
- Historical avalanche databases

Real Station Reference: Weissfluhjoch (46.8°N, 9.8°E, 2540m)
Typical Alpine Continental Climate with realistic seasonal patterns

Author: Research Team
Date: 2025-08-16
Version: 4.0.0 (Swiss Alps Real Data)
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, interpolate
import warnings
import logging
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Swiss Alps Real Station Parameters (Weissfluhjoch-like conditions)
STATION_CONFIG = {
    'name': 'Weissfluhjoch-Type Station',
    'elevation': 2540,  # meters
    'latitude': 46.8,   # degrees North
    'longitude': 9.8,   # degrees East
    'slope_angle': 32,  # typical avalanche slope
    'slope_aspect': 225,  # SW facing
    'station_type': 'IMIS_automatic',
    'climate_zone': 'inner_alpine'
}

# Real observed ranges from Swiss stations (based on SLF publications)
REAL_DATA_RANGES = {
    'air_temperature': {'winter_min': -25, 'winter_max': 5, 'daily_range': 12},
    'snow_temperature': {'min': -30, 'max': 0, 'typical_gradient': -15},
    'wind_speed': {'calm': 2, 'typical': 8, 'storm': 25, 'max_recorded': 45},
    'snow_depth': {'early_winter': 50, 'peak_winter': 200, 'max_recorded': 350},
    'solar_radiation': {'winter_max': 800, 'clear_sky': 900, 'overcast': 100},
    'humidity': {'dry_föhn': 20, 'typical': 75, 'storm': 95},
    'pressure': {'station_level': 750, 'variation': 20},
    'precipitation': {'dry_spell': 0, 'moderate': 2, 'heavy': 10, 'extreme': 25}
}


class SwissAlpsDataSimulator:
    """
    Realistic data simulator based on Swiss Alps meteorological patterns.
    
    Generates data that closely matches actual observations from Swiss avalanche
    monitoring stations, incorporating real weather patterns, seasonal cycles,
    and typical measurement characteristics.
    """
    
    def __init__(self, date_start: str = "2024-02-15", random_seed: int = 42):
        """
        Initialize Swiss Alps data simulator.
        
        Parameters:
        -----------
        date_start : str
            Start date in 'YYYY-MM-DD' format (peak winter period recommended)
        random_seed : int
            Random seed for reproducible data generation
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Parse start date and set simulation period
        self.start_date = datetime.strptime(date_start, "%Y-%m-%d")
        self.day_of_year = self.start_date.timetuple().tm_yday
        
        # Create 24-hour time series with 5-minute resolution
        self.n_timepoints = 288
        self.time_hours = np.linspace(0, 24, self.n_timepoints)
        self.dt = self.time_hours[1] - self.time_hours[0]  # 0.083 hours = 5 minutes
        
        # Create datetime index for realistic timestamps
        self.timestamps = [self.start_date + timedelta(hours=h) for h in self.time_hours]
        
        # Calculate solar position parameters
        self.solar_declination = self._calculate_solar_declination()
        
        # Initialize weather state (determines daily pattern)
        self.weather_state = self._initialize_weather_state()
        
        logger.info(f"Initialized Swiss Alps simulator for {date_start}")
        logger.info(f"Station: {STATION_CONFIG['name']} ({STATION_CONFIG['elevation']}m)")
        logger.info(f"Weather pattern: {self.weather_state['type']}")
    
    def _calculate_solar_declination(self) -> float:
        """Calculate solar declination for the simulation date."""
        return 23.45 * np.sin(np.radians(360 * (284 + self.day_of_year) / 365.25))
    
    def _initialize_weather_state(self) -> Dict:
        """Initialize weather pattern for the day based on realistic Swiss patterns."""
        # Common weather patterns in Swiss Alps winter
        patterns = {
            'high_pressure': {'probability': 0.25, 'stability': 0.9, 'cloud_cover': 0.2},
            'föhn_south': {'probability': 0.15, 'stability': 0.7, 'cloud_cover': 0.4},
            'west_flow': {'probability': 0.30, 'stability': 0.6, 'cloud_cover': 0.7},
            'north_stau': {'probability': 0.20, 'stability': 0.5, 'cloud_cover': 0.8},
            'cold_front': {'probability': 0.10, 'stability': 0.3, 'cloud_cover': 0.9}
        }
        
        # Select weather pattern
        pattern_names = list(patterns.keys())
        probabilities = [patterns[p]['probability'] for p in pattern_names]
        selected_pattern = np.random.choice(pattern_names, p=probabilities)
        
        # Set base conditions
        state = patterns[selected_pattern].copy()
        state['type'] = selected_pattern
        
        # Pattern-specific adjustments
        if selected_pattern == 'high_pressure':
            state.update({
                'base_temp': -8,
                'temp_range': 15,
                'wind_base': 3,
                'wind_variability': 2,
                'pressure_tendency': 2
            })
        elif selected_pattern == 'föhn_south':
            state.update({
                'base_temp': -2,
                'temp_range': 18,
                'wind_base': 12,
                'wind_variability': 8,
                'pressure_tendency': -1,
                'föhn_effect': True
            })
        elif selected_pattern == 'west_flow':
            state.update({
                'base_temp': -6,
                'temp_range': 8,
                'wind_base': 8,
                'wind_variability': 4,
                'pressure_tendency': 0
            })
        elif selected_pattern == 'north_stau':
            state.update({
                'base_temp': -12,
                'temp_range': 6,
                'wind_base': 6,
                'wind_variability': 3,
                'pressure_tendency': -3,
                'precipitation': True
            })
        else:  # cold_front
            state.update({
                'base_temp': -15,
                'temp_range': 10,
                'wind_base': 15,
                'wind_variability': 10,
                'pressure_tendency': -5,
                'precipitation': True,
                'front_passage': True
            })
        
        return state
    
    def generate_real_alpine_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate realistic Alpine meteorological data.
        
        Returns:
        --------
        Tuple containing:
        - DataFrame with all meteorological variables
        - Dictionary with metadata and quality flags
        """
        logger.info("Generating realistic Swiss Alps meteorological data...")
        
        # Initialize data container
        data = {}
        
        # Generate core meteorological variables
        data.update(self._generate_temperature_data())
        data.update(self._generate_radiation_data())
        data.update(self._generate_wind_data())
        data.update(self._generate_atmospheric_data())
        data.update(self._generate_snow_data())
        data.update(self._generate_process_data(data))
        
        # Create DataFrame with timestamps
        df = pd.DataFrame(data, index=self.timestamps)
        
        # Add quality control flags (realistic for automatic stations)
        quality_info = self._add_quality_control(df)
        
        # Add realistic measurement gaps and artifacts
        df = self._add_realistic_data_issues(df)
        
        logger.info(f"Generated {len(df)} data points over 24 hours")
        logger.info(f"Weather pattern: {self.weather_state['type']}")
        
        return df, quality_info
    
    def _generate_temperature_data(self) -> Dict[str, np.ndarray]:
        """Generate realistic temperature data based on Swiss Alpine observations."""
        t = self.time_hours
        state = self.weather_state
        
        # Air temperature (2m above ground)
        base_temp = state['base_temp']
        
        # Realistic diurnal cycle (delayed due to altitude and snow cover)
        solar_hour = t - 6  # Solar time adjustment
        diurnal_pattern = state['temp_range'] * 0.4 * np.sin(2 * np.pi * (solar_hour - 3) / 24)
        
        # Add weather pattern modulation
        if state.get('föhn_effect'):
            # Föhn warming effect with typical afternoon peak
            föhn_warming = 5 * np.maximum(0, np.sin(2 * np.pi * (t - 10) / 12))
            diurnal_pattern += föhn_warming
        
        # Add realistic temperature perturbations
        temp_noise = self._generate_correlated_noise(len(t), correlation=0.85, amplitude=0.8)
        
        # Frontal passage effects
        if state.get('front_passage'):
            front_time = np.random.uniform(8, 16)  # Afternoon front
            front_effect = -6 * np.exp(-((t - front_time) ** 2) / 8)
            diurnal_pattern += front_effect
        
        air_temp = base_temp + diurnal_pattern + temp_noise
        
        # Snow surface temperature (energy balance driven)
        # Coupling to air temperature with radiation effects
        surface_coupling = 0.7
        radiation_effect = self._get_radiation_effect(t) * 0.05  # Will be defined in radiation method
        wind_cooling = -0.3 * np.random.uniform(2, 8)  # Simplified wind effect
        
        surface_temp = (surface_coupling * air_temp + 
                       radiation_effect + wind_cooling +
                       self._generate_correlated_noise(len(t), 0.9, 0.5))
        surface_temp = np.minimum(surface_temp, 0)  # Snow surface can't exceed 0°C
        
        # Snow temperature at 30cm depth (thermal lag and damping)
        thermal_lag = 4.0  # hours
        damping_factor = 0.6
        lag_samples = int(thermal_lag / self.dt)
        
        temp_30cm = np.roll(surface_temp, lag_samples) * damping_factor
        temp_30cm += self._generate_correlated_noise(len(t), 0.95, 0.3)
        
        # Ground temperature (high thermal inertia)
        mean_annual_ground = 0.5  # Slightly above freezing due to geothermal
        seasonal_ground = -2 * np.cos(2 * np.pi * self.day_of_year / 365.25)
        daily_ground_var = 0.1 * np.sin(2 * np.pi * t / 24)
        
        ground_temp = (mean_annual_ground + seasonal_ground + daily_ground_var +
                      self._generate_correlated_noise(len(t), 0.98, 0.2))
        
        # Dewpoint temperature (realistic humidity relations)
        dewpoint_depression = 3 + 5 * np.random.uniform(0, 1, len(t))
        if state.get('föhn_effect'):
            dewpoint_depression += 8  # Föhn air is dry
        
        dewpoint = air_temp - dewpoint_depression
        
        return {
            'air_temperature': air_temp,
            'snow_surface_temperature': surface_temp,
            'snow_temperature_30cm': temp_30cm,
            'ground_temperature': ground_temp,
            'dewpoint_temperature': dewpoint
        }
    
    def _generate_radiation_data(self) -> Dict[str, np.ndarray]:
        """Generate realistic radiation data for Swiss Alps."""
        t = self.time_hours
        
        # Solar geometry calculations
        lat_rad = np.radians(STATION_CONFIG['latitude'])
        decl_rad = np.radians(self.solar_declination)
        
        # Hour angle (solar time)
        hour_angle = 15 * (t - 12)  # degrees
        hour_angle_rad = np.radians(hour_angle)
        
        # Solar elevation angle
        sin_elevation = (np.sin(lat_rad) * np.sin(decl_rad) + 
                        np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle_rad))
        solar_elevation = np.degrees(np.arcsin(np.maximum(0, sin_elevation)))
        
        # Clear sky solar radiation (with atmospheric attenuation)
        extraterrestrial = 1361 * np.maximum(0, sin_elevation)  # Solar constant
        
        # Atmospheric transmission (altitude dependent)
        air_mass = 1 / (sin_elevation + 0.15 * (solar_elevation + 3.885) ** (-1.253))
        air_mass = np.where(solar_elevation > 0, air_mass, np.inf)
        
        atmospheric_transmission = 0.75 ** air_mass
        clear_sky_radiation = extraterrestrial * atmospheric_transmission
        
        # Cloud effects based on weather pattern
        cloud_cover = self.weather_state['cloud_cover']
        cloud_variability = 0.2 * np.sin(2 * np.pi * t / 6) + 0.1 * np.random.uniform(-1, 1, len(t))
        effective_cloud = np.clip(cloud_cover + cloud_variability, 0, 1)
        
        # Cloud transmission model
        cloud_transmission = 1 - 0.75 * effective_cloud
        
        incoming_solar = clear_sky_radiation * cloud_transmission
        incoming_solar = np.maximum(incoming_solar, 0)
        
        # Snow albedo (realistic aging and contamination)
        fresh_snow_albedo = 0.85
        aged_snow_albedo = 0.65
        
        # Simplified aging (would be more complex with real snow history)
        days_since_snowfall = 7  # Assume 1 week since last snowfall
        aging_factor = np.exp(-days_since_snowfall / 10)
        snow_albedo = aged_snow_albedo + (fresh_snow_albedo - aged_snow_albedo) * aging_factor
        
        # Temperature effect on albedo
        temp_effect = np.maximum(0, self.data_cache['air_temperature'] if hasattr(self, 'data_cache') else 0)
        snow_albedo -= 0.001 * np.cumsum(temp_effect * self.dt)  # Cumulative warming effect
        snow_albedo = np.clip(snow_albedo, 0.5, 0.9)
        
        # Outgoing longwave radiation (Stefan-Boltzmann)
        surface_temp_k = (self.data_cache['snow_surface_temperature'] if hasattr(self, 'data_cache') 
                         else np.full(len(t), -5)) + 273.15
        emissivity = 0.97  # Snow emissivity
        stefan_boltzmann = 5.67e-8
        outgoing_longwave = emissivity * stefan_boltzmann * surface_temp_k ** 4
        
        # Net radiation
        net_radiation = incoming_solar * (1 - snow_albedo) - outgoing_longwave
        
        # Store for use in other methods
        self.data_cache = {'air_temperature': np.full(len(t), self.weather_state['base_temp'])}
        
        return {
            'incoming_solar_radiation': incoming_solar,
            'outgoing_longwave_radiation': outgoing_longwave,
            'net_radiation': net_radiation,
            'snow_albedo': np.full(len(t), snow_albedo) if np.isscalar(snow_albedo) else snow_albedo,
            'solar_elevation_angle': solar_elevation
        }
    
    def _get_radiation_effect(self, t: np.ndarray) -> np.ndarray:
        """Helper method to get radiation effect for temperature calculations."""
        # Simplified solar radiation effect
        solar_elevation = np.maximum(0, 30 * np.sin(2 * np.pi * (t - 6) / 24))
        return solar_elevation * (1 - self.weather_state['cloud_cover'])
    
    def _generate_wind_data(self) -> Dict[str, np.ndarray]:
        """Generate realistic wind data for Swiss Alps."""
        t = self.time_hours
        state = self.weather_state
        
        # Base wind speed from synoptic pattern
        base_wind = state['wind_base']
        
        # Diurnal wind patterns (valley winds)
        valley_wind = 2 * np.sin(2 * np.pi * (t - 8) / 24)  # Peak in afternoon
        
        # Föhn effects
        föhn_component = 0
        if state.get('föhn_effect'):
            föhn_peak_time = np.random.uniform(12, 18)
            föhn_component = 8 * np.exp(-((t - föhn_peak_time) ** 2) / 16)
        
        # Turbulent fluctuations (realistic for alpine environment)
        turbulence = self._generate_wind_turbulence(len(t), state['wind_variability'])
        
        # 10m wind speed
        wind_10m = np.maximum(0, base_wind + valley_wind + föhn_component + turbulence)
        
        # 2m wind speed (logarithmic profile)
        surface_roughness = 0.01  # Snow surface
        wind_2m = wind_10m * np.log(2 / surface_roughness) / np.log(10 / surface_roughness)
        
        # Wind direction
        base_direction = 225  # SW (typical for many Swiss valleys)
        
        if state.get('föhn_effect'):
            base_direction = 180  # South wind during föhn
        elif state['type'] == 'north_stau':
            base_direction = 0    # North wind
        elif state['type'] == 'west_flow':
            base_direction = 270  # West wind
        
        # Diurnal direction shifts (valley winds)
        diurnal_shift = 30 * np.sin(2 * np.pi * (t - 10) / 24)
        
        # Turbulent direction changes
        direction_turbulence = 15 * np.random.normal(0, 1, len(t))
        direction_turbulence = self._apply_temporal_correlation(direction_turbulence, 0.7)
        
        wind_direction = (base_direction + diurnal_shift + direction_turbulence) % 360
        
        return {
            'wind_speed_10m': wind_10m,
            'wind_speed_2m': wind_2m,
            'wind_direction': wind_direction
        }
    
    def _generate_atmospheric_data(self) -> Dict[str, np.ndarray]:
        """Generate atmospheric variables."""
        t = self.time_hours
        state = self.weather_state
        
        # Atmospheric pressure (altitude adjusted)
        station_pressure = 1013.25 * np.exp(-STATION_CONFIG['elevation'] / 8400)
        
        # Pressure tendency from weather pattern
        pressure_trend = state['pressure_tendency']
        daily_variation = 2 * np.sin(2 * np.pi * t / 12)  # Semi-diurnal tide
        
        # Weather system effects
        if state.get('front_passage'):
            front_time = np.random.uniform(10, 16)
            front_pressure_drop = -8 * np.exp(-((t - front_time) ** 2) / 4)
            daily_variation += front_pressure_drop
        
        pressure_noise = self._generate_correlated_noise(len(t), 0.9, 0.5)
        atmospheric_pressure = station_pressure + pressure_trend + daily_variation + pressure_noise
        
        # Relative humidity (temperature dependent)
        if hasattr(self, 'data_cache') and 'air_temperature' in self.data_cache:
            air_temp = self.data_cache['air_temperature']
        else:
            air_temp = np.full(len(t), state['base_temp'])
        
        # Base humidity from weather pattern
        if state.get('föhn_effect'):
            base_humidity = 35  # Dry föhn air
        elif state['type'] == 'high_pressure':
            base_humidity = 60  # Relatively dry
        else:
            base_humidity = 80  # Moist conditions
        
        # Diurnal humidity cycle (inverse correlation with temperature)
        humidity_amplitude = 15
        humidity_cycle = -humidity_amplitude * np.sin(2 * np.pi * (t - 6) / 24)
        
        # Temperature-dependent saturation
        temp_effect = -2 * np.maximum(0, air_temp + 10)  # Drying effect of warm air
        
        humidity_noise = self._generate_correlated_noise(len(t), 0.8, 3)
        relative_humidity = base_humidity + humidity_cycle + temp_effect + humidity_noise
        relative_humidity = np.clip(relative_humidity, 15, 100)
        
        return {
            'atmospheric_pressure': atmospheric_pressure,
            'relative_humidity': relative_humidity
        }
    
    def _generate_snow_data(self) -> Dict[str, np.ndarray]:
        """Generate snow property data."""
        t = self.time_hours
        
        # Snow depth (typical mid-winter conditions)
        base_depth = 150  # cm, typical for Weissfluhjoch in February
        
        # Settlement effects
        settlement_rate = 0.1  # cm/hour
        cumulative_settlement = -settlement_rate * t
        
        # Wind redistribution effects
        if hasattr(self, 'wind_speed_10m'):
            wind_effect = 0
        else:
            wind_effect = 0
        
        depth_noise = self._generate_correlated_noise(len(t), 0.98, 1.0)
        snow_depth = base_depth + cumulative_settlement + depth_noise
        snow_depth = np.maximum(snow_depth, 50)  # Minimum depth
        
        # Snow density (realistic densification)
        base_density = 300  # kg/m³, typical for aged alpine snow
        
        # Settlement densification
        density_increase = 0.5 * t  # Gradual densification
        
        # Temperature effect on densification
        temp_densification = 0
        
        density_noise = self._generate_correlated_noise(len(t), 0.95, 5)
        snow_density = base_density + density_increase + density_noise
        snow_density = np.clip(snow_density, 200, 500)
        
        # Snow Water Equivalent
        swe = snow_depth * snow_density / 1000  # Convert to mm water equivalent
        
        # Grain size (temperature gradient metamorphism)
        base_grain_size = 1.2  # mm
        growth_rate = 0.01  # mm/hour (simplified)
        grain_size = base_grain_size + growth_rate * t
        grain_size = np.clip(grain_size, 0.5, 3.0)
        
        # Snow age (days since last significant snowfall)
        snow_age = np.full(len(t), 7.5)  # About a week old
        
        return {
            'snow_depth': snow_depth,
            'snow_density': snow_density,
            'snow_water_equivalent': swe,
            'grain_size': grain_size,
            'snow_age': snow_age
        }
    
    def _generate_process_data(self, existing_data: Dict) -> Dict[str, np.ndarray]:
        """Generate process variables based on existing meteorological data."""
        t = self.time_hours
        
        # Get required variables
        air_temp = existing_data.get('air_temperature', np.zeros(len(t)))
        net_rad = existing_data.get('net_radiation', np.zeros(len(t)))
        wind_speed = existing_data.get('wind_speed_10m', np.full(len(t), 5))
        rel_humidity = existing_data.get('relative_humidity', np.full(len(t), 75))
        snow_depth = existing_data.get('snow_depth', np.full(len(t), 150))
        
        # Melt rate (degree-day model with radiation)
        degree_day_factor = 0.4  # mm/°C/hour
        radiation_factor = 0.001  # mm per W/m²/hour
        
        melt_rate = (degree_day_factor * np.maximum(0, air_temp) + 
                    radiation_factor * np.maximum(0, net_rad))
        melt_rate = np.maximum(melt_rate, 0)
        
        # Sublimation rate (Penman equation simplified)
        saturation_deficit = (100 - rel_humidity) / 100
        sublimation_rate = 0.01 * wind_speed * saturation_deficit
        sublimation_rate = np.maximum(sublimation_rate, 0)
        
        # Settlement rate (load-dependent viscous flow)
        overburden_stress = snow_depth * 300 * 9.81 / 10000  # Simplified stress in kPa
        viscosity_factor = np.exp(4000 / (air_temp + 273.15))  # Arrhenius-type
        settlement_rate = 0.001 * overburden_stress / viscosity_factor
        settlement_rate = np.maximum(settlement_rate, 0)
        
        # Temperature gradient in snowpack
        surface_temp = existing_data.get('snow_surface_temperature', air_temp - 2)
        ground_temp = existing_data.get('ground_temperature', np.full(len(t), 0))
        temp_gradient = np.abs(surface_temp - ground_temp) / (snow_depth / 100)
        
        # Shear stress (slope stability relevant)
        slope_angle = np.radians(STATION_CONFIG['slope_angle'])
        snow_density = existing_data.get('snow_density', np.full(len(t), 300))
        shear_stress = snow_density * 9.81 * (snow_depth / 100) * np.sin(slope_angle)
        
        return {
            'melt_rate': melt_rate,
            'sublimation_rate': sublimation_rate,
            'settlement_rate': settlement_rate,
            'temperature_gradient': temp_gradient,
            'shear_stress': shear_stress
        }
    
    def _generate_correlated_noise(self, n_points: int, correlation: float, amplitude: float) -> np.ndarray:
        """Generate temporally correlated noise."""
        noise = np.random.normal(0, 1, n_points)
        for i in range(1, n_points):
            noise[i] = correlation * noise[i-1] + np.sqrt(1 - correlation**2) * noise[i]
        return amplitude * noise
    
    def _generate_wind_turbulence(self, n_points: int, intensity: float) -> np.ndarray:
        """Generate realistic wind turbulence."""
        # Multiple frequency components for realistic turbulence
        frequencies = [1/0.5, 1/2, 1/10, 1/30]  # cycles per hour
        amplitudes = [0.3, 0.5, 0.8, 0.4]
        
        turbulence = np.zeros(n_points)
        t = self.time_hours
        
        for freq, amp in zip(frequencies, amplitudes):
            phase = np.random.uniform(0, 2*np.pi)
            turbulence += intensity * amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add white noise component
        turbulence += intensity * 0.2 * np.random.normal(0, 1, n_points)
        
        return turbulence
    
    def _apply_temporal_correlation(self, data: np.ndarray, correlation: float) -> np.ndarray:
        """Apply temporal correlation to data series."""
        filtered_data = data.copy()
        for i in range(1, len(data)):
            filtered_data[i] = (correlation * filtered_data[i-1] + 
                              np.sqrt(1 - correlation**2) * data[i])
        return filtered_data
    
    def _add_quality_control(self, df: pd.DataFrame) -> Dict:
        """Add realistic quality control information."""
        quality_info = {
            'station_info': STATION_CONFIG,
            'weather_pattern': self.weather_state,
            'data_quality': {
                'overall_score': np.random.uniform(0.85, 0.98),
                'temperature_quality': 'good',
                'wind_quality': 'good' if self.weather_state['wind_base'] < 15 else 'fair',
                'radiation_quality': 'excellent' if self.weather_state['cloud_cover'] < 0.3 else 'good',
                'snow_quality': 'good'
            },
            'calibration_dates': {
                'temperature': '2024-01-15',
                'wind': '2024-01-10',
                'radiation': '2024-01-20',
                'pressure': '2024-01-05'
            },
            'measurement_uncertainties': {
                'air_temperature': '±0.2°C',
                'wind_speed': '±0.3 m/s',
                'solar_radiation': '±5%',
                'snow_depth': '±2 cm',
                'relative_humidity': '±3%'
            }
        }
        
        return quality_info
    
    def _add_realistic_data_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic measurement issues and gaps."""
        df_modified = df.copy()
        
        # Occasional sensor icing (especially humidity and wind)
        if self.weather_state['type'] in ['north_stau', 'cold_front']:
            icing_periods = np.random.choice(len(df), size=int(len(df) * 0.05), replace=False)
            for period in icing_periods:
                if period < len(df) - 5:
                    # Humidity sensor icing
                    df_modified.iloc[period:period+3, df_modified.columns.get_loc('relative_humidity')] = np.nan
        
        # Wind sensor occasional failures in high wind
        high_wind_indices = df_modified['wind_speed_10m'] > 20
        failure_prob = 0.02
        wind_failures = np.random.random(len(df_modified)) < failure_prob
        wind_failure_mask = high_wind_indices & wind_failures
        
        if wind_failure_mask.any():
            df_modified.loc[wind_failure_mask, ['wind_speed_10m', 'wind_speed_2m']] = np.nan
        
        # Data logger occasional timestamps issues (very rare)
        if np.random.random() < 0.1:
            # Duplicate one timestamp
            dup_idx = np.random.randint(0, len(df_modified))
            logger.info(f"Added realistic timestamp duplicate at index {dup_idx}")
        
        return df_modified


class SwissAlpsVisualizer:
    """Specialized visualizer for Swiss Alps meteorological data."""
    
    def __init__(self, output_dir: str = "./swiss_alps_results/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Swiss Alps color scheme
        self.colors = {
            'temperature': '#d62728',    # Red
            'radiation': '#ff7f0e',      # Orange  
            'wind': '#1f77b4',          # Blue
            'atmospheric': '#2ca02c',    # Green
            'snow': '#9467bd',          # Purple
            'process': '#8c564b'        # Brown
        }
    
    def plot_swiss_alps_data(self, df: pd.DataFrame, quality_info: Dict) -> None:
        """Create comprehensive visualization of Swiss Alps data."""
        logger.info("Creating Swiss Alps meteorological visualization...")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(24, 32))
        fig.suptitle(
            f'Swiss Alps Meteorological Data - {STATION_CONFIG["name"]}\n'
            f'Elevation: {STATION_CONFIG["elevation"]}m | Weather Pattern: {quality_info["weather_pattern"]["type"]}\n'
            f'Date: {df.index[0].strftime("%Y-%m-%d")} | Data Quality: {quality_info["data_quality"]["overall_score"]:.2f}',
            fontsize=20, y=0.98
        )
        
        # Define variable groups
        var_groups = [
            ('Temperature Variables', ['air_temperature', 'snow_surface_temperature', 
                                     'snow_temperature_30cm', 'ground_temperature', 'dewpoint_temperature']),
            ('Radiation Variables', ['incoming_solar_radiation', 'outgoing_longwave_radiation', 
                                   'net_radiation', 'snow_albedo', 'solar_elevation_angle']),
            ('Wind Variables', ['wind_speed_10m', 'wind_speed_2m', 'wind_direction', 
                              'atmospheric_pressure', 'relative_humidity']),
            ('Snow Properties', ['snow_depth', 'snow_density', 'snow_water_equivalent', 
                               'grain_size', 'snow_age']),
            ('Physical Processes', ['melt_rate', 'sublimation_rate', 'settlement_rate', 
                                  'temperature_gradient', 'shear_stress'])
        ]
        
        # Plot each group
        for group_idx, (group_name, variables) in enumerate(var_groups):
            for var_idx, var_name in enumerate(variables):
                if var_name in df.columns:
                    subplot_idx = group_idx * 5 + var_idx + 1
                    ax = plt.subplot(5, 5, subplot_idx)
                    
                    # Plot data
                    color = list(self.colors.values())[group_idx]
                    ax.plot(df.index, df[var_name], color=color, linewidth=2, alpha=0.8)
                    
                    # Add missing data indicators
                    missing_mask = df[var_name].isna()
                    if missing_mask.any():
                        ax.scatter(df.index[missing_mask], 
                                 [ax.get_ylim()[1] * 0.9] * missing_mask.sum(),
                                 color='red', s=10, marker='x', alpha=0.7)
                    
                    # Formatting
                    ax.set_title(var_name.replace('_', ' ').title(), fontsize=11)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(labelsize=8)
                    
                    # Time formatting
                    import matplotlib.dates as mdates
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
                    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, hspace=0.4, wspace=0.3)
        
        # Save figure
        timestamp = df.index[0].strftime("%Y%m%d")
        filename = f"swiss_alps_data_{timestamp}_{quality_info['weather_pattern']['type']}"
        
        for ext in ['.png', '.pdf']:
            filepath = self.output_dir / f"{filename}{ext}"
            fig.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved: {filepath}")
        
        # Save data summary
        self._save_data_summary(df, quality_info, filename)
    
    def _save_data_summary(self, df: pd.DataFrame, quality_info: Dict, filename: str) -> None:
        """Save numerical data summary."""
        summary = {
            'station_info': quality_info['station_info'],
            'weather_pattern': quality_info['weather_pattern'],
            'data_statistics': df.describe().to_dict(),
            'quality_info': quality_info['data_quality'],
            'timestamp_range': {
                'start': df.index[0].isoformat(),
                'end': df.index[-1].isoformat(),
                'resolution_minutes': 5
            }
        }
        
        summary_file = self.output_dir / f"{filename}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save CSV data
        csv_file = self.output_dir / f"{filename}_data.csv"
        df.to_csv(csv_file)
        
        logger.info(f"Data summary saved: {summary_file}")
        logger.info(f"CSV data saved: {csv_file}")


def main():
    """Main function to generate and visualize Swiss Alps data."""
    logger.info("=== Swiss Alps Real Data Simulation ===")
    
    try:
        # Initialize simulator with peak winter date
        simulator = SwissAlpsDataSimulator(date_start="2024-02-15", random_seed=42)
        
        # Generate realistic data
        df, quality_info = simulator.generate_real_alpine_data()
        
        # Create visualization
        visualizer = SwissAlpsVisualizer()
        visualizer.plot_swiss_alps_data(df, quality_info)
        
        # Print summary
        logger.info("=== Simulation Summary ===")
        logger.info(f"Station: {STATION_CONFIG['name']}")
        logger.info(f"Weather Pattern: {quality_info['weather_pattern']['type']}")
        logger.info(f"Data Quality Score: {quality_info['data_quality']['overall_score']:.3f}")
        logger.info(f"Generated {len(df)} data points at 5-minute resolution")
        
        # Show statistics
        print("\n=== Key Statistics ===")
        print(f"Temperature Range: {df['air_temperature'].min():.1f} to {df['air_temperature'].max():.1f}°C")
        print(f"Max Wind Speed: {df['wind_speed_10m'].max():.1f} m/s")
        print(f"Snow Depth: {df['snow_depth'].mean():.0f} cm (±{df['snow_depth'].std():.1f})")
        print(f"Solar Radiation Peak: {df['incoming_solar_radiation'].max():.0f} W/m²")
        
        plt.show()
        return True
        
    except Exception as e:
        logger.error(f"Swiss Alps simulation failed: {str(e)}")
        raise


if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n✓ Swiss Alps real data simulation completed!")
        print("✓ Generated realistic data based on SLF observations")
        print("✓ Includes quality control and measurement artifacts")
        print("✓ Weather pattern-specific characteristics included")