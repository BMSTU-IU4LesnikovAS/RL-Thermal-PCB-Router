"""All physical assumptions in one place. Every value below is an input
assumption (not a measurement) and is reported in the paper's Table I."""
import math

MU0 = 4 * math.pi * 1e-7

# Power stage
P_NOM_W = 50e3
V_DC = 800.0
I_LOAD = P_NOM_W / V_DC          # 62.5 A
DI_DT = 5e9                      # A/s (5 A/ns)
C_OSS = 150e-12                  # F, effective output capacitance
R_LOOP = 0.5                     # ohm, damping resistance at ringing frequency
L_PACKAGE_NH = 5.0               # nH, device + DC-link capacitor ESL

# Board
BOARD_MM = 80.0
T_CU = 105e-6                    # m, 3 oz trace layer
T_PLANE = 35e-6                  # m, 1 oz return plane
T_FR4 = 1.6e-3                   # m
H_DIEL = 0.2e-3                  # m, trace-to-return-plane dielectric
K_CU = 385.0                     # W/m/K
K_FR4 = 0.3                      # W/m/K
RHO_CU = 2.2e-8                  # ohm*m at about 85 C

# Thermal path
T_AMB_C = 25.0
R_SA = 0.25                      # K/W, heat sink to ambient
R_TIM = 5e-4                     # m2*K/W
H_BOARD = 1.0 / (T_FR4 / K_FR4 + R_TIM)      # W/m2/K, through the bare board
H_DEVICE = 3000.0                # W/m2/K, under devices (thermal-via array)
P_DEVICE_W = 20.0                # W per MOSFET

# Trace width levels (mm) and footprint half-width in cells for a 2.5 mm cell
WIDTHS_MM = (2.5, 7.5, 12.5)

# Fixed layout of the case study, in board fractions (row0, row1, col0, col1)
DEVICES = ((0.25, 0.4375, 0.40625, 0.59375),
           (0.5625, 0.75, 0.40625, 0.59375))
SOURCE = (0.5, 0.03125)
TARGET = (0.5, 0.9375)

# Normalisation constants for the reward (dimensionless terms)
L_REF_NH = 10.0
T_REF_C = 100.0
