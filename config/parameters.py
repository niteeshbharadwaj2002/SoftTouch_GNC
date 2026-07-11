# config/params.py

# --- Environment ---
G = 9.81          # gravity (m/s^2)
G0 = 9.81         # standard gravity for Isp->mdot conversion

# --- Rocket / Propulsion ---
ISP = 282         # specific impulse (s)
DRY_MASS = 1500.0     # kg, mass without fuel
FUEL_MASS = 500.0     # kg, initial propellant mass
WET_MASS = DRY_MASS + FUEL_MASS   # kg, total initial mass

MAX_THRUST = 24000.0   # N, per-engine or total max thrust
MIN_THRUST = 6000.0    # N, minimum throttle (engines can't go to zero)

# --- Initial Conditions ---
INITIAL_ALTITUDE = 1500.0   # m
INITIAL_DOWNRANGE = 500.0   # m
INITIAL_VX = -30.0           # m/s (toward target)
INITIAL_VY = -80.0           # m/s (descending, negative = downward)

INITIAL_STATE = [
    INITIAL_DOWNRANGE,   # x
    INITIAL_ALTITUDE,    # y
    INITIAL_VX,          # vx
    INITIAL_VY,           # vy
    WET_MASS,             # m
]

# --- Target / Landing Conditions ---
TARGET_X = 0.0        # downrange position of pad
TARGET_Y = 0.0        # ground level
TARGET_VX = 0.0        # soft landing: zero velocity
TARGET_VY = 0.0

# --- Simulation ---
DT = 0.1              # timestep (s)
MAX_SIM_TIME = 60.0   # s, safety cap
N_STEPS = int(MAX_SIM_TIME / DT)