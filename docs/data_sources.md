# Data sources and required acknowledgements

All the datasets are openly licensed. The Copernicus Marine datasets need a free account. The dataset IDs
below were checked against the Copernicus Marine catalogue with `copernicusmarine` 2.4.1 in September 2026.

## Western Mediterranean (`configs/wmed.yaml`)

| Role | Product | Dataset ID | Variable | Grid |
|---|---|---|---|---|
| Truth SSH | MEDSEA_MULTIYEAR_PHY_006_004 | `cmems_mod_med_phy-ssh_my_4.2km_P1D-m` | `zos` | 1/24° |
| Truth SST | MEDSEA_MULTIYEAR_PHY_006_004 | `cmems_mod_med_phy-temp_my_4.2km_P1D-m` | `thetao` (top level) | 1/24° |
| L4 ADT | SEALEVEL_EUR_PHY_L4_MY_008_068 | `cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.0625deg_P1D` | `adt` | 1/16° |
| L4 SST | SST_MED_SST_L4_REP_OBSERVATIONS_010_021 | `cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021` | `analysed_sst` | 0.05° |

## Black Sea (`configs/blacksea.yaml`)

| Role | Product | Dataset ID | Variable | Grid |
|---|---|---|---|---|
| Truth SSH | BLKSEA_MULTIYEAR_PHY_007_004 | `cmems_mod_blk_phy-ssh_my_2.5km_P1D-m` | `zos` | 1/40° |
| Truth SST | BLKSEA_MULTIYEAR_PHY_007_004 | `cmems_mod_blk_phy-temp_my_2.5km_P1D-m` | `thetao` | 1/40° |
| L4 ADT | SEALEVEL_EUR_PHY_L4_MY_008_068 | `cmems_obs-sl_eur_phy-ssh_my_allsat-l4-duacs-0.0625deg_P1D` | `adt` | 1/16° |
| L4 SST | SST_BS_SST_L4_REP_OBSERVATIONS_010_022 | `cmems_SST_BS_SST_L4_REP_OBSERVATIONS_010_022` | `analysed_sst` | 0.05° |

## In-situ drifters

The **NOAA Global Drifter Program** 6-hourly quality-controlled dataset is fetched through the AOML ERDDAP
server at `https://erddap.aoml.noaa.gov/gdp/erddap/tabledap/drifter_6hour_qc`. It uses the variables
`ID, time, latitude, longitude, ve, vn, drogue_lost_date`.

Coverage I checked in September 2026 (6-hourly observations):

| Box | Years | Drifters | Observations |
|---|---|---|---|
| Alboran–Balearic, 2016–2024 | 2016–2024 | 298 | 80,431 |
| Black Sea, 1993–2025 | only 2001–04 and 2009–10 | 32 | 17,752 |

If you need more Black Sea drifters, the Copernicus in-situ TAC products can be added. Save them as
`data/<region>/drifters.csv` with the columns `id,time,lat,lon,u,v`.

## SWOT (optional comparison)

SWOT L3 KaRIn products are available from AVISO+ and from NASA PO.DAAC. The 2022–2024 reconstruction
period overlaps the SWOT science orbit, which began in July 2023. That makes a direct comparison with
2-D swath SSH at about 2 km possible as a future extension.

## Acknowledgement text for the paper and poster

> This study has been conducted using E.U. Copernicus Marine Service Information
> (doi of each product as listed on its catalogue page). Drifter data were provided by the NOAA Global
> Drifter Program (Lumpkin & Centurioni, 2019, NOAA NCEI, doi:10.25921/7ntx-z961).
